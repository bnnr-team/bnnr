"""Lightweight k-fold cross-validation utilities for bnnr analyze (classification).

This operates on cached predictions and labels from a single evaluation run.
It does NOT retrain the model; instead, it measures how global metrics vary
across folds of the validation set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from bnnr.analysis.schema import CrossValidationResults


def _make_stratified_folds(labels: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Create stratified folds indices for 1D integer labels."""
    rng = np.random.default_rng(0)
    labels = labels.astype(int)
    classes = np.unique(labels)
    fold_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            fold_indices[i % n_folds].append(int(idx))
    return [np.asarray(sorted(f), dtype=int) for f in fold_indices]


@dataclass
class _FoldMetrics:
    fold: int
    accuracy: float
    support: int
    per_class_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cross_validation_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
) -> CrossValidationResults:
    """Compute simple k-fold CV metrics from per-sample predictions and labels."""
    if n_folds < 2:
        return CrossValidationResults(n_folds=0, global_metrics={}, per_fold_metrics=[])
    if preds.shape[0] != labels.shape[0]:
        raise ValueError("preds and labels must have the same length for CV.")

    preds = preds.astype(int)
    labels = labels.astype(int)
    folds = _make_stratified_folds(labels, n_folds)
    all_fold_metrics: list[_FoldMetrics] = []
    accs: list[float] = []

    classes = np.unique(labels)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    k = len(classes)

    for fold_id, fold_idx in enumerate(folds):
        if fold_idx.size == 0:
            continue
        y_true = labels[fold_idx]
        y_pred = preds[fold_idx]
        conf = np.zeros((k, k), dtype=int)
        for t, p in zip(y_true, y_pred):
            ti = class_to_idx[int(t)]
            pi = class_to_idx[int(p)]
            conf[ti, pi] += 1
        total = int(conf.sum())
        acc = float(np.trace(conf) / total) if total > 0 else 0.0
        accs.append(acc)

        per_class_recall: dict[str, float] = {}
        for cls_val, row_idx in class_to_idx.items():
            row = conf[row_idx]
            support = int(row.sum())
            rec = float(row[row_idx] / support) if support > 0 else 0.0
            per_class_recall[str(cls_val)] = rec

        all_fold_metrics.append(
            _FoldMetrics(
                fold=fold_id,
                accuracy=acc,
                support=int(fold_idx.size),
                per_class_recall=per_class_recall,
            )
        )

    if not accs:
        return CrossValidationResults(n_folds=0, global_metrics={}, per_fold_metrics=[])

    global_metrics = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
    }
    per_fold_dicts = [fm.to_dict() for fm in all_fold_metrics]
    return CrossValidationResults(
        n_folds=len(all_fold_metrics),
        global_metrics=global_metrics,
        per_fold_metrics=per_fold_dicts,
    )

