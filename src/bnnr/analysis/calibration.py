"""Top-1 confidence calibration metrics for multiclass classification (analyze)."""

from __future__ import annotations

import numpy as np


def compute_top1_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    """Expected Calibration Error using max-probability bins (multiclass).

    Bins are uniform on [0, 1]. ECE = sum_b (|B_b|/n) * |acc_b - conf_b|.

    Parameters
    ----------
    confidences
        Predicted probability of the predicted class, shape ``(N,)``, in [0, 1].
    correct
        Whether the top-1 prediction matched the label, shape ``(N,)`` bool or 0/1.
    """
    if confidences.size == 0:
        return {
            "ece": 0.0,
            "n_bins": float(n_bins),
            "mean_confidence": 0.0,
            "accuracy": 0.0,
        }
    conf = np.clip(confidences.astype(np.float64), 0.0, 1.0)
    corr = correct.astype(np.bool_).astype(np.float64)
    n = float(conf.size)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_b = float(corr[mask].mean())
        conf_b = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)

    return {
        "ece": round(float(ece), 6),
        "n_bins": float(n_bins),
        "mean_confidence": round(float(conf.mean()), 6),
        "accuracy": round(float(corr.mean()), 6),
    }
