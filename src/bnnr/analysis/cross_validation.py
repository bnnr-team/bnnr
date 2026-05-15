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
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cohen_kappa: float
    support: int
    per_class_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cohen_kappa_from_matrix(mat: np.ndarray) -> float:
    """Compute Cohen's Kappa from a confusion matrix."""
    total = float(mat.sum())
    if total == 0:
        return 0.0
    p_o = float(np.trace(mat)) / total
    row_sums = mat.sum(axis=1).astype(float)
    col_sums = mat.sum(axis=0).astype(float)
    p_e = float((row_sums * col_sums).sum()) / (total * total)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def _compute_fold_metrics(conf: np.ndarray, k: int) -> tuple[float, float, float, float, float]:
    """Compute accuracy, precision_macro, recall_macro, f1_macro, cohen_kappa from confusion matrix."""
    total = int(conf.sum())
    acc = float(np.trace(conf) / total) if total > 0 else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    for i in range(k):
        tp = float(conf[i, i])
        support_i = float(conf[i, :].sum())
        pred_i = float(conf[:, i].sum())
        precisions.append(tp / pred_i if pred_i > 0 else 0.0)
        recalls.append(tp / support_i if support_i > 0 else 0.0)

    prec_macro = float(np.mean(precisions)) if precisions else 0.0
    rec_macro = float(np.mean(recalls)) if recalls else 0.0
    f1s = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precisions, recalls)
    ]
    f1_macro = float(np.mean(f1s)) if f1s else 0.0
    kappa = _cohen_kappa_from_matrix(conf)
    return acc, prec_macro, rec_macro, f1_macro, kappa


def run_cross_validation_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
) -> CrossValidationResults:
    """Compute k-fold CV metrics from per-sample predictions and labels."""
    if n_folds < 2:
        return CrossValidationResults(n_folds=0, global_metrics={}, per_fold_metrics=[])
    if preds.shape[0] != labels.shape[0]:
        raise ValueError("preds and labels must have the same length for CV.")

    preds = preds.astype(int)
    labels = labels.astype(int)
    folds = _make_stratified_folds(labels, n_folds)
    all_fold_metrics: list[_FoldMetrics] = []
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    kappas: list[float] = []

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

        acc, prec_m, rec_m, f1_m, kappa = _compute_fold_metrics(conf, k)
        accs.append(acc)
        precs.append(prec_m)
        recs.append(rec_m)
        f1s.append(f1_m)
        kappas.append(kappa)

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
                precision_macro=prec_m,
                recall_macro=rec_m,
                f1_macro=f1_m,
                cohen_kappa=kappa,
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
        "mean_precision_macro": float(np.mean(precs)),
        "std_precision_macro": float(np.std(precs)),
        "mean_recall_macro": float(np.mean(recs)),
        "std_recall_macro": float(np.std(recs)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "mean_cohen_kappa": float(np.mean(kappas)),
        "std_cohen_kappa": float(np.std(kappas)),
    }
    per_fold_dicts = [fm.to_dict() for fm in all_fold_metrics]
    return CrossValidationResults(
        n_folds=len(all_fold_metrics),
        global_metrics=global_metrics,
        per_fold_metrics=per_fold_dicts,
    )


def _macro_f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over labels; shapes (N, L) with 0/1 values."""
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        return 0.0
    _, L = y_true.shape
    f1s: list[float] = []
    for j in range(L):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _subset_accuracy_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return 0.0
    return float(np.mean(np.all(y_true.astype(int) == y_pred.astype(int), axis=1)))


def run_cross_validation_multilabel_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
) -> CrossValidationResults:
    """K-fold metrics on multi-label predictions (same split for all labels)."""
    if n_folds < 2 or preds.shape[0] != labels.shape[0]:
        return CrossValidationResults(n_folds=0, global_metrics={}, per_fold_metrics=[])
    n = preds.shape[0]
    rng = np.random.default_rng(0)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    f1_macros: list[float] = []
    sub_accs: list[float] = []
    per_fold_dicts: list[dict[str, Any]] = []

    for fold_id, fold_idx in enumerate(folds):
        if fold_idx.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[fold_idx] = False
        te = fold_idx
        y_t = labels[te]
        y_p = preds[te]
        f1m = _macro_f1_multilabel(y_t, y_p)
        sa = _subset_accuracy_multilabel(y_t, y_p)
        f1_macros.append(f1m)
        sub_accs.append(sa)
        per_fold_dicts.append({
            "fold": fold_id,
            "f1_macro": round(f1m, 4),
            "subset_accuracy": round(sa, 4),
            "support": int(te.size),
        })

    if not f1_macros:
        return CrossValidationResults(n_folds=0, global_metrics={}, per_fold_metrics=[])

    global_metrics = {
        "mean_f1_macro": float(np.mean(f1_macros)),
        "std_f1_macro": float(np.std(f1_macros)),
        "mean_subset_accuracy": float(np.mean(sub_accs)),
        "std_subset_accuracy": float(np.std(sub_accs)),
    }
    return CrossValidationResults(
        n_folds=len(per_fold_dicts),
        global_metrics=global_metrics,
        per_fold_metrics=per_fold_dicts,
    )

