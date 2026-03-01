"""Detection metrics for BNNR.

Computes mAP@0.5, mAP@[.50:.95], and per-class AP using a lightweight
implementation based on the PASCAL VOC / COCO evaluation protocol.

Falls back to ``torchmetrics.detection.MeanAveragePrecision`` when
available, otherwise uses a bundled pure-PyTorch implementation that
covers the most common use cases.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _compute_iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Parameters
    ----------
    boxes1 : Tensor[N, 4]
    boxes2 : Tensor[M, 4]

    Returns
    -------
    Tensor[N, M] – IoU matrix.
    """
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def _voc_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AP using 101-point interpolation (COCO style)."""
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    precision_interp = np.zeros_like(recall_thresholds)

    for i, t in enumerate(recall_thresholds):
        precisions_at_recall = precision[recall >= t]
        if len(precisions_at_recall) > 0:
            precision_interp[i] = float(precisions_at_recall.max())

    return float(precision_interp.mean())


def _compute_ap_single_class(
    per_image_pred_boxes: list[Tensor],
    per_image_pred_scores: list[Tensor],
    per_image_gt_boxes: list[Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP for a single class at a given IoU threshold.

    Matching is done **per-image** to prevent cross-image false matches.
    All inputs are lists where index *i* corresponds to image *i*.

    Parameters
    ----------
    per_image_pred_boxes : list[Tensor]
        ``[Tensor[Ni, 4], ...]`` predicted boxes per image (xyxy).
    per_image_pred_scores : list[Tensor]
        ``[Tensor[Ni], ...]`` confidence scores per image.
    per_image_gt_boxes : list[Tensor]
        ``[Tensor[Mi, 4], ...]`` ground-truth boxes per image (xyxy).
    iou_threshold : float
        Minimum IoU to consider a match.
    """
    n_gt = sum(len(g) for g in per_image_gt_boxes)

    if n_gt == 0:
        # No ground-truth: any prediction is a false positive → AP = 0
        has_preds = any(len(p) > 0 for p in per_image_pred_boxes)
        return 0.0 if has_preds else 1.0

    # Build flat list of (score, image_idx, local_pred_idx)
    all_entries: list[tuple[float, int, int]] = []
    for img_idx, (boxes, scores) in enumerate(
        zip(per_image_pred_boxes, per_image_pred_scores)
    ):
        for local_idx in range(len(scores)):
            all_entries.append((float(scores[local_idx].item()), img_idx, local_idx))

    if not all_entries:
        return 0.0

    # Sort globally by confidence (descending)
    all_entries.sort(key=lambda x: x[0], reverse=True)

    # Per-image matched-GT tracking
    matched: list[Tensor] = [
        torch.zeros(len(g), dtype=torch.bool) for g in per_image_gt_boxes
    ]

    tp = np.zeros(len(all_entries))
    fp = np.zeros(len(all_entries))

    for i, (_, img_idx, local_idx) in enumerate(all_entries):
        gt_boxes = per_image_gt_boxes[img_idx]
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        pred_box = per_image_pred_boxes[img_idx][local_idx : local_idx + 1]
        ious = _compute_iou_matrix(pred_box, gt_boxes)[0]

        best_gt = int(ious.argmax().item())
        best_iou = float(ious[best_gt].item())

        if best_iou >= iou_threshold and not matched[img_idx][best_gt]:
            tp[i] = 1
            matched[img_idx][best_gt] = True
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (n_gt + 1e-8)

    return _voc_ap(precision, recall)


def calculate_detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_thresholds: list[float] | None = None,
    score_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute detection metrics across a full epoch.

    Parameters
    ----------
    predictions : list[dict]
        Each dict has ``boxes`` (Tensor[N, 4]), ``scores`` (Tensor[N]),
        ``labels`` (Tensor[N]).  Boxes in xyxy format.
    targets : list[dict]
        Each dict has ``boxes`` (Tensor[M, 4]), ``labels`` (Tensor[M]).
        Boxes in xyxy format.
    iou_thresholds : list[float] | None
        IoU thresholds for AP computation. Default: [0.5, 0.55, ..., 0.95].
    score_threshold : float
        Minimum score to consider a prediction.

    Returns
    -------
    dict with keys: map_50, map_50_95, per_class_ap (nested dict).
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    # Try torchmetrics first (most accurate, COCO-compatible)
    try:
        return _calculate_with_torchmetrics(predictions, targets, iou_thresholds, score_threshold)
    except (ImportError, Exception):
        pass

    # Fallback: built-in implementation
    return _calculate_builtin(predictions, targets, iou_thresholds, score_threshold)


def _calculate_with_torchmetrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_thresholds: list[float],
    score_threshold: float,
) -> dict[str, float]:
    """Use torchmetrics MeanAveragePrecision for COCO-compatible evaluation."""
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)

    # Filter by score threshold
    filtered_preds = []
    for pred in predictions:
        mask = pred["scores"] >= score_threshold
        filtered_preds.append({
            "boxes": pred["boxes"][mask],
            "scores": pred["scores"][mask],
            "labels": pred["labels"][mask],
        })

    metric.update(filtered_preds, targets)
    result = metric.compute()

    return {
        "map_50": float(result["map_50"].item()),
        "map_50_95": float(result["map"].item()),
    }


def _calculate_builtin(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_thresholds: list[float],
    score_threshold: float,
) -> dict[str, float]:
    """Built-in AP computation (no torchmetrics dependency).

    Matching is done **per-image** (COCO-style) to avoid cross-image
    false matches that inflate or deflate AP.
    """
    # Collect all unique class IDs
    all_classes: set[int] = set()
    for target in targets:
        if len(target["labels"]) > 0:
            all_classes.update(target["labels"].cpu().tolist())
    for pred in predictions:
        if len(pred["labels"]) > 0:
            all_classes.update(pred["labels"].cpu().tolist())

    if not all_classes:
        return {"map_50": 0.0, "map_50_95": 0.0}

    # Per-class, per-threshold AP
    ap_values_50: list[float] = []
    ap_values_all: list[float] = []

    for class_id in sorted(all_classes):
        # Gather per-image pred/gt boxes for this class
        per_img_pred_boxes: list[Tensor] = []
        per_img_pred_scores: list[Tensor] = []
        per_img_gt_boxes: list[Tensor] = []

        for pred, target in zip(predictions, targets):
            # Predictions for this class in this image
            pred_mask = pred["labels"] == class_id
            if pred["scores"].numel() > 0:
                score_mask = pred["scores"] >= score_threshold
                pred_mask = pred_mask & score_mask

            if pred_mask.any():
                per_img_pred_boxes.append(pred["boxes"][pred_mask].cpu())
                per_img_pred_scores.append(pred["scores"][pred_mask].cpu())
            else:
                per_img_pred_boxes.append(torch.zeros(0, 4))
                per_img_pred_scores.append(torch.zeros(0))

            # Ground truth for this class in this image
            gt_mask = target["labels"] == class_id
            if gt_mask.any():
                per_img_gt_boxes.append(target["boxes"][gt_mask].cpu())
            else:
                per_img_gt_boxes.append(torch.zeros(0, 4))

        class_aps: list[float] = []
        for iou_t in iou_thresholds:
            ap = _compute_ap_single_class(
                per_img_pred_boxes, per_img_pred_scores, per_img_gt_boxes, iou_t,
            )
            class_aps.append(ap)

        ap_50 = class_aps[0] if class_aps else 0.0
        ap_mean = float(np.mean(class_aps)) if class_aps else 0.0

        ap_values_50.append(ap_50)
        ap_values_all.append(ap_mean)

    return {
        "map_50": float(np.mean(ap_values_50)) if ap_values_50 else 0.0,
        "map_50_95": float(np.mean(ap_values_all)) if ap_values_all else 0.0,
    }


def calculate_per_class_ap(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-class AP at a given IoU threshold.

    Matching is done **per-image** (consistent with ``_calculate_builtin``).

    Returns a dict mapping class_id (str) -> {"ap": float, "support": int, "name": str}.
    """
    all_classes: set[int] = set()
    for target in targets:
        if len(target["labels"]) > 0:
            all_classes.update(target["labels"].cpu().tolist())

    result: dict[str, dict[str, Any]] = {}
    for class_id in sorted(all_classes):
        per_img_pred_boxes: list[Tensor] = []
        per_img_pred_scores: list[Tensor] = []
        per_img_gt_boxes: list[Tensor] = []

        total_gt = 0
        for pred, target in zip(predictions, targets):
            pred_mask = pred["labels"] == class_id
            if pred_mask.any():
                per_img_pred_boxes.append(pred["boxes"][pred_mask].cpu())
                per_img_pred_scores.append(pred["scores"][pred_mask].cpu())
            else:
                per_img_pred_boxes.append(torch.zeros(0, 4))
                per_img_pred_scores.append(torch.zeros(0))

            gt_mask = target["labels"] == class_id
            if gt_mask.any():
                gt_boxes = target["boxes"][gt_mask].cpu()
                per_img_gt_boxes.append(gt_boxes)
                total_gt += len(gt_boxes)
            else:
                per_img_gt_boxes.append(torch.zeros(0, 4))

        ap = _compute_ap_single_class(
            per_img_pred_boxes, per_img_pred_scores, per_img_gt_boxes, iou_threshold,
        )
        name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"

        result[str(class_id)] = {
            "ap": ap,
            "support": total_gt,
            "name": name,
        }

    return result


__all__ = [
    "calculate_detection_metrics",
    "calculate_per_class_ap",
    "calculate_detection_confusion_matrix",
]


def calculate_detection_confusion_matrix(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    num_classes: int | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a class-level confusion matrix for detection.

    Matrix semantics:
    - rows: true class (GT)
    - cols: predicted class
    - TP cross-class confusions are counted by IoU matching:
      for each GT, take the highest-IoU unmatched prediction (if IoU >= threshold).
    - Unmatched GT contributes to background prediction column (0).
    - Unmatched predictions contribute to background true row (0).

    Class id 0 is reserved for background.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    all_classes: set[int] = {0}
    for target in targets:
        if len(target.get("labels", [])) > 0:
            all_classes.update(int(x) for x in target["labels"].cpu().tolist())
    for pred in predictions:
        if len(pred.get("labels", [])) > 0:
            all_classes.update(int(x) for x in pred["labels"].cpu().tolist())

    if num_classes is not None:
        labels = list(range(max(1, num_classes)))
        if 0 not in labels:
            labels = [0] + labels
    else:
        labels = sorted(all_classes)
    index_of = {cls: i for i, cls in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for pred, target in zip(predictions, targets):
        p_boxes = pred.get("boxes", torch.zeros((0, 4)))
        p_labels = pred.get("labels", torch.zeros((0,), dtype=torch.long))
        g_boxes = target.get("boxes", torch.zeros((0, 4)))
        g_labels = target.get("labels", torch.zeros((0,), dtype=torch.long))

        if not isinstance(p_boxes, Tensor):
            p_boxes = torch.as_tensor(p_boxes, dtype=torch.float32)
        if not isinstance(p_labels, Tensor):
            p_labels = torch.as_tensor(p_labels, dtype=torch.long)
        if not isinstance(g_boxes, Tensor):
            g_boxes = torch.as_tensor(g_boxes, dtype=torch.float32)
        if not isinstance(g_labels, Tensor):
            g_labels = torch.as_tensor(g_labels, dtype=torch.long)

        if p_boxes.numel() == 0 and g_boxes.numel() == 0:
            continue

        pred_used = torch.zeros((len(p_boxes),), dtype=torch.bool)

        # Match each GT to best unmatched prediction
        for gi in range(len(g_boxes)):
            gt_cls = int(g_labels[gi].item()) if gi < len(g_labels) else 0
            if len(p_boxes) == 0:
                matrix[index_of.get(gt_cls, 0), index_of[0]] += 1
                continue
            ious = _compute_iou_matrix(g_boxes[gi : gi + 1], p_boxes)[0]
            if pred_used.any():
                ious = ious.masked_fill(pred_used, -1.0)
            best_pi = int(ious.argmax().item())
            best_iou = float(ious[best_pi].item())
            if best_iou >= iou_threshold:
                pred_cls = int(p_labels[best_pi].item()) if best_pi < len(p_labels) else 0
                matrix[index_of.get(gt_cls, 0), index_of.get(pred_cls, 0)] += 1
                pred_used[best_pi] = True
            else:
                # Missed GT -> predicted as background
                matrix[index_of.get(gt_cls, 0), index_of[0]] += 1

        # Remaining unmatched predictions -> background true row
        for pi in range(len(p_boxes)):
            if pred_used[pi]:
                continue
            pred_cls = int(p_labels[pi].item()) if pi < len(p_labels) else 0
            matrix[index_of[0], index_of.get(pred_cls, 0)] += 1

    return {"labels": labels, "matrix": matrix.tolist()}
