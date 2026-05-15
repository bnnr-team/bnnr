"""Per-class diagnostics: precision, recall, F1, support, true/pred distributions.

Used by analyze to rank critical classes and detect over/under-prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bnnr.analysis.schema import ClassDiagnostic


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


def compute_class_diagnostics(
    confusion: dict[str, Any],
    *,
    n_classes: int | None = None,
) -> tuple[list[ClassDiagnostic], dict[str, int], dict[str, int]]:
    """Compute per-class precision, recall, F1, Cohen's Kappa and distributions.

    confusion must have "matrix" (list of lists) and "labels" (list of class ids).
    Returns (list of ClassDiagnostic, true_distribution, pred_distribution).
    """
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    if not isinstance(matrix, list) or not matrix or not labels_list:
        return [], {}, {}

    mat = np.asarray(matrix, dtype=np.int64)
    n = mat.shape[0]
    if n_classes is not None:
        n = min(n, n_classes)

    true_dist: dict[str, int] = {}
    pred_dist: dict[str, int] = {}
    diagnostics: list[ClassDiagnostic] = []

    for i in range(n):
        class_id = str(labels_list[i]) if i < len(labels_list) else str(i)
        support = int(mat[i, :].sum())
        pred_as_i = int(mat[:, i].sum())
        true_dist[class_id] = support
        pred_dist[class_id] = pred_as_i

        tp = int(mat[i, i])
        recall = tp / support if support > 0 else 0.0
        precision = tp / pred_as_i if pred_as_i > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = tp / support if support > 0 else 0.0

        binary = np.zeros((2, 2), dtype=np.int64)
        binary[0, 0] = tp
        binary[0, 1] = support - tp
        binary[1, 0] = pred_as_i - tp
        binary[1, 1] = int(mat.sum()) - support - pred_as_i + tp
        kappa = _cohen_kappa_from_matrix(binary)

        severity = "ok"
        if recall <= 0 and support > 0:
            severity = "critical"
        elif recall < 0.5 or precision < 0.5:
            severity = "warning"

        diagnostics.append(
            ClassDiagnostic(
                class_id=class_id,
                accuracy=round(acc, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=support,
                pred_count=pred_as_i,
                cohen_kappa=round(kappa, 4),
                severity=severity,
            )
        )

    # Rank by severity then by F1 (worst first)
    def rank_key(d: ClassDiagnostic) -> tuple[int, float]:
        sev_order = {"critical": 0, "warning": 1, "ok": 2}
        return (sev_order.get(d.severity, 2), -d.f1)

    diagnostics.sort(key=rank_key)
    for r, d in enumerate(diagnostics, start=1):
        d.rank = r

    return diagnostics, true_dist, pred_dist


def build_distribution_summary(
    true_dist: dict[str, int],
    pred_dist: dict[str, int],
) -> dict[str, Any]:
    """Return a summary suitable for report: over/under-predicted classes, collapse hint."""
    total_true = sum(true_dist.values())
    total_pred = sum(pred_dist.values())
    if total_true == 0:
        return {"true_total": 0, "pred_total": 0, "over_predicted": [], "under_predicted": []}

    over: list[dict[str, Any]] = []
    under: list[dict[str, Any]] = []
    for cid in set(true_dist) | set(pred_dist):
        t = true_dist.get(cid, 0)
        p = pred_dist.get(cid, 0)
        if t > 0 and p > t * 1.2:
            over.append({"class": cid, "true": t, "pred": p, "ratio": round(p / t, 2)})
        if t > 0 and p < t * 0.8:
            under.append({"class": cid, "true": t, "pred": p, "ratio": round(p / t, 2)})

    over.sort(key=lambda x: -x["ratio"])
    under.sort(key=lambda x: x["ratio"])
    return {
        "true_total": total_true,
        "pred_total": total_pred,
        "over_predicted": over[:10],
        "under_predicted": under[:10],
        "possible_collapse": (
            len(pred_dist) < len(true_dist)
            and max(pred_dist.values() or [0]) > total_pred * 0.5
        ),
    }


def compute_global_cohen_kappa(confusion: dict[str, Any]) -> float:
    """Compute global Cohen's Kappa from a confusion matrix dict."""
    matrix = confusion.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        return 0.0
    return _cohen_kappa_from_matrix(np.asarray(matrix, dtype=np.int64))


def compute_multilabel_label_diagnostics(
    confusion: dict[str, Any],
) -> tuple[list[ClassDiagnostic], dict[str, int], dict[str, int]]:
    """Per-label binary diagnostics from ``multilabel_per_label`` confusion.

    ``accuracy`` is per-label accuracy (TP+TN)/N. Cohen's kappa is the 2×2
    kappa for that label vs rest.
    """
    if confusion.get("type") != "multilabel_per_label":
        return [], {}, {}
    labels_list = confusion.get("labels", [])
    per_label = confusion.get("per_label", [])
    n_samples = int(confusion.get("n_samples", 0))
    if not isinstance(per_label, list) or not labels_list or n_samples <= 0:
        return [], {}, {}

    true_dist: dict[str, int] = {}
    pred_dist: dict[str, int] = {}
    diagnostics: list[ClassDiagnostic] = []

    for i, cell in enumerate(per_label):
        if not isinstance(cell, dict):
            continue
        class_id = str(labels_list[i]) if i < len(labels_list) else str(i)
        tp = int(cell.get("tp", 0))
        fp = int(cell.get("fp", 0))
        fn = int(cell.get("fn", 0))
        tn = n_samples - tp - fp - fn
        if tn < 0:
            tn = 0
        support = tp + fn
        pred_as_pos = tp + fp
        true_dist[class_id] = support
        pred_dist[class_id] = pred_as_pos

        prec = tp / pred_as_pos if pred_as_pos > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / n_samples if n_samples > 0 else 0.0

        # Rows/cols: negative vs positive (true × pred) for Cohen's kappa
        binary = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        kappa = _cohen_kappa_from_matrix(binary)

        severity = "ok"
        if support > 0 and rec <= 0:
            severity = "critical"
        elif support > 0 and (rec < 0.5 or prec < 0.5):
            severity = "warning"

        diagnostics.append(
            ClassDiagnostic(
                class_id=class_id,
                accuracy=round(acc, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                support=support,
                pred_count=pred_as_pos,
                cohen_kappa=round(kappa, 4),
                severity=severity,
            )
        )

    def rank_key(d: ClassDiagnostic) -> tuple[int, float]:
        sev_order = {"critical": 0, "warning": 1, "ok": 2}
        return (sev_order.get(d.severity, 2), -d.f1)

    diagnostics.sort(key=rank_key)
    for r, d in enumerate(diagnostics, start=1):
        d.rank = r

    return diagnostics, true_dist, pred_dist
