"""Extended metrics for the grand benchmark.

All functions take a PyTorch model/loader and return plain ``dict[str, float]``.
No sklearn required — implemented from scratch with PyTorch + numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute accuracy, f1_macro, top5_accuracy, cohen_kappa, ece.

    Parameters
    ----------
    model:
        PyTorch model in eval mode (called with model.eval() internally).
    loader:
        DataLoader whose batches are ``(images, labels, ...)``.
    num_classes:
        Number of output classes.
    device:
        Torch device string.
    top_k:
        k for top-k accuracy; skipped if ``num_classes < top_k``.

    Returns
    -------
    dict with keys: accuracy, f1_macro, top5_accuracy (if applicable),
    cohen_kappa, ece.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            # Batch may be (images, labels) or (images, labels, indices)
            imgs = batch[0].to(device)
            labels = batch[1]
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
            all_probs.append(probs)

    if not all_preds:
        return {"accuracy": 0.0, "f1_macro": 0.0, "cohen_kappa": 0.0, "ece": 0.0}

    preds_t = torch.cat(all_preds)          # (N,)
    labels_t = torch.cat(all_labels)        # (N,)
    probs_t = torch.cat(all_probs)          # (N, C)
    n_total = len(labels_t)

    # --- accuracy ---
    correct = (preds_t == labels_t).sum().item()
    accuracy = correct / n_total

    # --- f1_macro ---
    f1_macro = _f1_macro(preds_t, labels_t, num_classes)

    # --- top-k accuracy ---
    result: dict[str, float] = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }
    if num_classes >= top_k:
        topk_correct = _topk_correct(probs_t, labels_t, k=top_k)
        result["top5_accuracy"] = float(topk_correct / n_total)

    # --- cohen_kappa ---
    result["cohen_kappa"] = float(_cohen_kappa(preds_t, labels_t, num_classes))

    # --- ECE (10 bins) ---
    result["ece"] = float(_ece(probs_t, labels_t, n_bins=10))

    return result


def _f1_macro(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 from per-class precision and recall."""
    f1_scores: list[float] = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _topk_correct(probs: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Count top-k correct predictions."""
    top_k_preds = probs.topk(k, dim=1).indices  # (N, k)
    labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded).any(dim=1).sum().item()
    return int(correct)


def _cohen_kappa(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e)."""
    n = len(labels)
    if n == 0:
        return 0.0
    p_o = (preds == labels).sum().item() / n  # observed agreement

    # p_e = sum over classes of (P(actual=k) * P(predicted=k))
    p_e = 0.0
    for c in range(num_classes):
        p_actual = (labels == c).sum().item() / n
        p_predicted = (preds == c).sum().item() / n
        p_e += p_actual * p_predicted

    if (1.0 - p_e) < 1e-8:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    confidences, preds = probs.max(dim=1)
    correct = (preds == labels).float()

    conf_np = confidences.numpy()
    correct_np = correct.numpy()

    ece_val = 0.0
    n_total = len(conf_np)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf_np >= lo) & (conf_np < hi)
        if i == n_bins - 1:  # include right edge for last bin
            mask = (conf_np >= lo) & (conf_np <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc_bin = float(correct_np[mask].mean())
        conf_bin = float(conf_np[mask].mean())
        ece_val += (n_bin / n_total) * abs(conf_bin - acc_bin)

    return float(ece_val)


# ---------------------------------------------------------------------------
# XAI metrics
# ---------------------------------------------------------------------------


def compute_xai_metrics(
    adapter: Any,
    loader: DataLoader,
    sample_indices: list[int],
    device: str,
    method: str = "opticam",
) -> dict[str, float]:
    """Compute XAI quality metrics over a set of sample indices.

    Parameters
    ----------
    adapter:
        A ``SimpleTorchAdapter``-compatible object with ``get_model()`` and
        ``get_target_layers()`` methods.
    loader:
        DataLoader used to retrieve images by dataset index.
    sample_indices:
        Dataset-level indices of the images to analyse.
    device:
        Torch device string.
    method:
        XAI method passed to ``generate_saliency_maps`` (default: opticam).

    Returns
    -------
    dict with keys: edge_ratio, coverage, gini_coefficient, entropy,
    center_bias.  Returns all-zero dict if no valid samples.
    """
    from bnnr.xai import generate_saliency_maps
    from bnnr.xai_analysis import analyze_saliency_map

    empty = {
        "xai_edge_ratio": 0.0,
        "xai_coverage": 0.0,
        "xai_gini": 0.0,
        "xai_entropy": 0.0,
        "xai_center_bias": 0.0,
    }

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return empty

    images_t: list[torch.Tensor] = []
    labels_t: list[torch.Tensor] = []
    used_indices: list[int] = []

    for ds_idx in sample_indices:
        try:
            sample = dataset[ds_idx]
        except (IndexError, KeyError):
            continue
        img = sample[0] if isinstance(sample, (list, tuple)) else sample
        lbl = sample[1] if isinstance(sample, (list, tuple)) and len(sample) > 1 else 0
        if not isinstance(img, torch.Tensor):
            continue
        images_t.append(img.unsqueeze(0))
        lbl_val = int(lbl.item()) if isinstance(lbl, torch.Tensor) else int(lbl)
        labels_t.append(torch.tensor(lbl_val).view(1))
        used_indices.append(ds_idx)

    if not images_t:
        return empty

    images_batch = torch.cat(images_t, dim=0)
    labels_batch = torch.cat(labels_t, dim=0)
    model = adapter.get_model()
    target_layers = adapter.get_target_layers()
    model_device = next(model.parameters()).device
    model.eval()

    try:
        maps = generate_saliency_maps(
            model,
            images_batch.to(model_device),
            labels_batch.to(model_device),
            target_layers,
            method=method,
        )
    except Exception:
        return empty

    per_sample: list[dict[str, float]] = []
    center_biases: list[float] = []
    for map_2d in maps:
        stats = analyze_saliency_map(map_2d)
        per_sample.append(stats)
        center_biases.append(_center_bias(map_2d))

    def _mean(key: str) -> float:
        vals = [float(s.get(key, 0.0)) for s in per_sample]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "xai_edge_ratio": _mean("edge_ratio"),
        "xai_coverage": _mean("coverage"),
        "xai_gini": _mean("gini"),
        "xai_entropy": _mean("entropy"),
        "xai_center_bias": float(np.mean(center_biases)) if center_biases else 0.0,
    }


def _center_bias(map_2d: np.ndarray) -> float:
    """Fraction of total saliency mass in the central 50% of the image."""
    h, w = map_2d.shape
    total = float(map_2d.sum())
    if total <= 1e-8:
        return 0.0
    # Central 50%: crop out the outer 25% on each side
    h_lo = h // 4
    h_hi = h - h // 4
    w_lo = w // 4
    w_hi = w - w // 4
    center_mass = float(map_2d[h_lo:h_hi, w_lo:w_hi].sum())
    return center_mass / total
