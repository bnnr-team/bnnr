"""Standalone evaluation logic for BNNR — reused by BNNRTrainer and analyze.

Provides run_evaluation (metrics + per_class + confusion) and
collect_eval_predictions (per-sample preds, labels, indices, confidences, losses)
for classification and multi-label, used by the training loop and by bnnr analyze
without duplicating eval logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from bnnr.adapter import XAICapableModel


def _average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: float(sum(m[k] for m in all_metrics) / len(all_metrics)) for k in keys}


def run_evaluation(
    adapter: Any,
    loader: DataLoader,
    config: Any,
    *,
    custom_metrics: dict[str, Any] | None = None,
    return_preds_labels: bool = False,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Run evaluation on the given adapter and loader.

    Returns (metrics, per_class_accuracy, confusion, preds, labels).
    For detection, preds and labels are None. For classification, preds/labels
    are 1-D when ``return_preds_labels`` is True. For multi-label, preds/labels
    are 2-D (``N × L``) when ``return_preds_labels`` is True.
    """
    task = getattr(config, "task", "classification")
    is_detection = task == "detection"
    is_multilabel = task == "multilabel"
    metrics_list = getattr(config, "metrics", ["accuracy", "f1_macro", "loss"])
    metrics_list = [m for m in metrics_list if m != "loss"]

    if is_detection:
        m, p, c, _, _ = _run_evaluation_detection(adapter, loader, config)
        return m, p, c, None, None
    if is_multilabel:
        m, p, c, preds, labels = _run_evaluation_multilabel(
            adapter, loader, config, metrics_list, custom_metrics
        )
        if not return_preds_labels:
            preds, labels = None, None
        return m, p, c, preds, labels
    m, p, c, preds, labels = _run_evaluation_classification(
        adapter, loader, config, metrics_list, custom_metrics
    )
    if not return_preds_labels:
        preds, labels = None, None
    return m, p, c, preds, labels


def _run_evaluation_classification(
    adapter: Any,
    loader: DataLoader,
    config: Any,
    metrics_list: list[str],
    custom_metrics: dict[str, Any] | None,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
]:
    can_cache = isinstance(adapter, XAICapableModel)
    all_metrics: list[dict[str, float]] = []
    preds_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    _captured_logits: list[torch.Tensor] = []
    _hook_handle = None
    if can_cache:
        model_impl = adapter.get_model()
        model_impl.eval()

        def _capture_hook(_module: Any, _inp: Any, output: Any) -> None:
            _captured_logits.append(output.detach())

        _hook_handle = model_impl.register_forward_hook(_capture_hook)

    try:
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
                batch: Any = (images, labels)
            else:
                batch = raw_batch
                images, labels = raw_batch[0], raw_batch[1]
            _captured_logits.clear()
            all_metrics.append(adapter.eval_step(batch))
            if can_cache and _captured_logits:
                logits = _captured_logits[-1]
                preds_rows.append(torch.argmax(logits, dim=1).cpu())
                label_rows.append(labels.cpu())
    finally:
        if _hook_handle is not None:
            _hook_handle.remove()

    result_metrics = _average_metrics(all_metrics)
    last_eval_preds = torch.cat(preds_rows).numpy().astype(np.int64) if preds_rows else None
    last_eval_labels = torch.cat(label_rows).numpy().astype(np.int64) if label_rows else None

    if custom_metrics and last_eval_preds is not None and last_eval_labels is not None:
        for name, fn in custom_metrics.items():
            try:
                result_metrics[name] = float(fn(last_eval_preds, last_eval_labels))
            except Exception:
                pass

    if last_eval_preds is None or last_eval_labels is None:
        return result_metrics, {}, {}, None, None

    preds = last_eval_preds
    labels = last_eval_labels
    n_classes = int(max(int(np.max(preds)), int(np.max(labels)))) + 1
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in range(n_classes):
        mask = labels == class_id
        support = int(np.sum(mask))
        if support == 0:
            continue
        acc = float(np.mean(preds[mask] == labels[mask]))
        per_class[str(class_id)] = {"accuracy": acc, "support": support}
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
        matrix[int(true_label), int(pred_label)] += 1
    confusion: dict[str, Any] = {
        "labels": list(range(n_classes)),
        "matrix": matrix.tolist(),
    }
    return result_metrics, per_class, confusion, preds, labels


def _run_evaluation_multilabel(
    adapter: Any,
    loader: DataLoader,
    config: Any,
    metrics_list: list[str],
    custom_metrics: dict[str, Any] | None,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Always capture logits via forward hook so per-label confusion is defined."""
    all_metrics: list[dict[str, float]] = []
    preds_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    threshold = getattr(config, "multilabel_threshold", 0.5)

    model_impl = adapter.get_model()
    model_impl.eval()
    _captured_logits: list[torch.Tensor] = []

    def _capture_hook(_module: Any, _inp: Any, output: Any) -> None:
        _captured_logits.append(output.detach())

    _hook_handle = model_impl.register_forward_hook(_capture_hook)

    try:
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
                batch = (images, labels)
            else:
                batch = raw_batch
                images, labels = raw_batch[0], raw_batch[1]
            _captured_logits.clear()
            all_metrics.append(adapter.eval_step(batch))
            if _captured_logits:
                logits = _captured_logits[-1]
                preds_rows.append((torch.sigmoid(logits) >= threshold).int().cpu())
                label_rows.append(labels.cpu())
    finally:
        _hook_handle.remove()

    result_metrics = _average_metrics(all_metrics)
    last_eval_preds = torch.cat(preds_rows).numpy() if preds_rows else None
    last_eval_labels = torch.cat(label_rows).numpy() if label_rows else None

    if custom_metrics and last_eval_preds is not None and last_eval_labels is not None:
        for name, fn in custom_metrics.items():
            try:
                result_metrics[name] = float(fn(last_eval_preds, last_eval_labels))
            except Exception:
                pass

    if last_eval_preds is None or last_eval_labels is None:
        return result_metrics, {}, {}, None, None

    preds = last_eval_preds
    labels = last_eval_labels
    n_samples = int(preds.shape[0])
    n_labels = preds.shape[1] if preds.ndim == 2 else 0
    if n_labels == 0:
        return result_metrics, {}, {}, None, None

    per_label: dict[str, dict[str, float | int]] = {}
    confusion_per_label: list[dict[str, int]] = []
    for label_idx in range(n_labels):
        y_true = labels[:, label_idx]
        y_pred = preds[:, label_idx]
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        support = int(np.sum(y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[str(label_idx)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        confusion_per_label.append({"tp": tp, "fp": fp, "fn": fn})
    confusion: dict[str, Any] = {
        "type": "multilabel_per_label",
        "labels": list(range(n_labels)),
        "per_label": confusion_per_label,
        "n_samples": n_samples,
    }
    return result_metrics, per_label, confusion, preds, labels


def _run_evaluation_detection(
    adapter: Any,
    loader: DataLoader,
    config: Any,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], dict[str, Any], None, None]:
    from bnnr.detection_metrics import (
        calculate_detection_confusion_matrix,
        calculate_per_class_ap,
    )

    for raw_batch in loader:
        if len(raw_batch) == 3:
            images, targets, _ = raw_batch
        else:
            images, targets = raw_batch
        batch = (images, targets)
        adapter.eval_step(batch)
    epoch_end_eval = getattr(adapter, "epoch_end_eval", None)
    if not callable(epoch_end_eval):
        return {}, {}, {}, None, None
    result_metrics = epoch_end_eval()
    all_preds = getattr(adapter, "last_eval_preds", [])
    all_targets = getattr(adapter, "last_eval_targets", [])
    if not all_preds or not all_targets:
        return result_metrics, {}, {}, None, None

    class_names = getattr(config, "detection_class_names", None)
    known_classes: set[int] = set()
    for t in all_targets:
        if t.get("labels") is not None and len(t["labels"]) > 0:
            known_classes.update(int(x) for x in t["labels"].cpu().tolist())

    per_class_ap = calculate_per_class_ap(all_preds, all_targets, class_names=class_names)
    per_class: dict[str, dict[str, float | int]] = {}
    for cls_id, info in per_class_ap.items():
        per_class[cls_id] = {
            "accuracy": info["ap"],
            "ap_50": info["ap"],
            "support": info["support"],
        }
    num_classes = None
    if class_names is not None:
        first_lower = (
            str(class_names[0]).strip().lower() if class_names else ""
        )
        if first_lower in {"background", "bg", "__background__"}:
            num_classes = len(class_names)
        else:
            num_classes = len(class_names) + 1
    confusion = calculate_detection_confusion_matrix(
        predictions=all_preds,
        targets=all_targets,
        num_classes=num_classes,
        iou_threshold=0.5,
    )
    if known_classes and confusion.get("labels"):
        allowed = known_classes | {0}
        old_labels = confusion["labels"]
        old_matrix = confusion["matrix"]
        keep_indices = [i for i, lbl in enumerate(old_labels) if lbl in allowed]
        if len(keep_indices) < len(old_labels):
            new_labels = [old_labels[i] for i in keep_indices]
            new_matrix = [
                [old_matrix[r][c] for c in keep_indices]
                for r in keep_indices
            ]
            confusion = {"labels": new_labels, "matrix": new_matrix}
    return result_metrics, per_class, confusion, None, None


def collect_eval_predictions(
    adapter: Any,
    loader: DataLoader,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect per-sample predictions, labels, indices, confidences and losses.

    Supports ``classification`` and ``multilabel``. Requires a model with
    ``get_model()`` and a ``criterion`` (same as training).
    Returns (preds, labels, indices, confidences, losses) or None if unavailable.
    """
    task = getattr(config, "task", "classification")
    if task == "detection":
        return None
    model = adapter.get_model()
    criterion = getattr(adapter, "criterion", None)
    if criterion is None:
        return None
    device = next(model.parameters()).device
    model.eval()

    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    indices_list: list[np.ndarray] = []
    confidences_list: list[np.ndarray] = []
    losses_list: list[np.ndarray] = []
    sample_offset = 0
    threshold = getattr(config, "multilabel_threshold", 0.5)

    with torch.no_grad():
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, labels_batch, indices_batch = raw_batch
                idx = (
                    indices_batch.cpu().numpy()
                    if isinstance(indices_batch, Tensor)
                    else np.asarray(indices_batch)
                )
            else:
                images, labels_batch = raw_batch
                batch_size = labels_batch.shape[0]
                idx = np.arange(sample_offset, sample_offset + batch_size)
                sample_offset += batch_size
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(images)

            if task == "multilabel":
                labels_batch = labels_batch.float()
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).int().cpu().numpy()
                conf, _ = probs.max(dim=1)
                if hasattr(criterion, "reduction"):
                    old_red = getattr(criterion, "reduction", "mean")
                    try:
                        criterion.reduction = "none"  # type: ignore[assignment]
                        loss_per = criterion(logits, labels_batch)
                    finally:
                        criterion.reduction = old_red  # type: ignore[assignment]
                else:
                    loss_per = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels_batch, reduction="none"
                    )
                if loss_per.ndim == 2:
                    loss_scalar = loss_per.mean(dim=1)
                else:
                    loss_scalar = loss_per.flatten()
                loss_np = loss_scalar.cpu().numpy()
            else:
                if labels_batch.ndim > 1 and labels_batch.shape[-1] == 1:
                    labels_batch = labels_batch.squeeze(-1)
                if hasattr(criterion, "reduction"):
                    old_red = getattr(criterion, "reduction", "mean")
                    try:
                        criterion.reduction = "none"  # type: ignore[assignment]
                        loss_per = criterion(logits, labels_batch)
                    finally:
                        criterion.reduction = old_red  # type: ignore[assignment]
                else:
                    loss_per = torch.nn.functional.cross_entropy(
                        logits, labels_batch.long(), reduction="none"
                    )
                probs = torch.softmax(logits, dim=1)
                conf, preds = probs.max(dim=1)
                preds = preds.cpu().numpy()
                loss_np = loss_per.cpu().numpy()

            preds_list.append(preds)
            labels_list.append(labels_batch.cpu().numpy())
            indices_list.append(idx)
            confidences_list.append(conf.cpu().numpy())
            losses_list.append(loss_np)

    if not preds_list:
        return None
    return (
        np.concatenate(preds_list),
        np.concatenate(labels_list),
        np.concatenate(indices_list),
        np.concatenate(confidences_list),
        np.concatenate(losses_list),
    )
