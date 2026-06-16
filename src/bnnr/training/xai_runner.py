"""XAI generation and reporting callbacks extracted from BNNRTrainer."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from bnnr.adapter import XAICapableModel
from bnnr.augmentations import BaseAugmentation
from bnnr.icd import AICD, ICD
from bnnr.training import callbacks as _callbacks
from bnnr.training import image_utils as _img
from bnnr.training import probe as _probe
from bnnr.training.checkpoint import _is_ultralytics_tasks_backbone
from bnnr.utils import lazy_cv2 as cv2
from bnnr.xai import generate_saliency_maps, save_xai_visualization
from bnnr.xai_analysis import analyze_xai_batch_rich
from bnnr.xai_cache import XAICache


def generate_xai(trainer, iteration: int, augmentation_name: str, confusion: dict[str, Any] | None = None) -> tuple[list[Path], dict[str, str], dict[str, dict[str, Any]]]:
    """Generate XAI saliency maps and per-class textual insights.

    Parameters
    ----------
    confusion : dict | None
        Confusion matrix dict (``{"labels": [...], "matrix": [[...]]}``).
        When provided, insights are enriched with confusion-pair info and
        cross-checkpoint trend indicators.

    Returns
    -------
    paths : list[Path]
        Saved overlay PNG paths.
    insights : dict[str, str]
        Mapping ``class_id`` → human-readable insight text.
    diagnoses : dict[str, dict]
        Structured per-class diagnosis (severity, quality_score, …).
        Empty dict when XAI is disabled.
    """
    if not trainer._xai_enabled:
        return [], {}, {}

    if trainer._is_detection:
        model_getter = getattr(trainer.model, "get_model", None)
        if not callable(model_getter):
            return [], {}, {}

        from bnnr.detection_xai import (
            compute_detection_box_saliency_occlusion,
            generate_detection_saliency,
            save_detection_xai_panels,
        )

        _probe.initialize_report_probe_samples(trainer)
        if trainer._report_probe_images is None or trainer._report_probe_targets is None:
            return [], {}, {}

        model_impl = model_getter()
        device = next(model_impl.parameters()).device
        predict_ultra = getattr(trainer.model, "predict_detection_dicts", None)
        use_ultra_xai = callable(predict_ultra)
        if _is_ultralytics_tasks_backbone(model_impl) and not use_ultra_xai:
            trainer._log(
                "Detection XAI skipped: Ultralytics backbone requires "
                "UltralyticsDetectionAdapter.predict_detection_dicts (raw task module only)."
            )
            return [], {}, {}

        # Generate XAI for ALL probe samples (activation method is
        # a single batched forward pass, so the cost is negligible).
        images = trainer._report_probe_images.to(device)
        targets = trainer._report_probe_targets

        xai_method = trainer.config.detection_xai_method
        forward_layout: Literal["torchvision_list", "ultralytics_bchw"] = (
            "ultralytics_bchw" if use_ultra_xai else "torchvision_list"
        )

        with torch.no_grad():
            if use_ultra_xai and predict_ultra is not None:
                preds = predict_ultra(images)
            else:
                preds = model_impl([img for img in images])

        # ── Activation-based saliency (fast: single batched forward pass) ──
        activation_saliency: list[np.ndarray] | None = None
        if xai_method == "activation":
            target_layers_fn = getattr(trainer.model, "get_target_layers", None)
            target_layers = target_layers_fn() if callable(target_layers_fn) else []
            if target_layers:
                activation_saliency = generate_detection_saliency(
                    model_impl,
                    images,
                    target_layers,
                    device=device,
                    forward_layout=forward_layout,
                )

        np_images = _img.tensor_batch_to_preview_uint8(images.detach().cpu())
        save_dir = trainer.config.report_dir / "xai" / f"iter_{iteration}_{augmentation_name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        class_names = trainer.config.detection_class_names
        paths: list[Path] = []
        trainer._latest_detection_xai_details = []
        per_class_scores: dict[str, list[float]] = defaultdict(list)

        ultra_predict_chw: Callable[[Tensor], dict[str, Tensor]] | None = None
        if use_ultra_xai and predict_ultra is not None:
            _pu = predict_ultra

            def _ultra_predict_chw(im: Tensor) -> dict[str, Tensor]:
                d = _pu(im.unsqueeze(0))[0]
                return {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in d.items()
                }

            ultra_predict_chw = _ultra_predict_chw

        for idx in range(np_images.shape[0]):
            gt = targets[idx]
            pred = preds[idx]
            pred_scores = pred.get("scores", torch.zeros(0))
            pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
            # Keep top-k predictions for readability
            top_k = min(5, int(pred_scores.numel()))
            if top_k > 0:
                top_idx = torch.argsort(pred_scores, descending=True)[:top_k]
                pred_boxes = pred["boxes"][top_idx].detach().cpu()
                pred_labels = pred_labels[top_idx].detach().cpu()
                pred_scores = pred_scores[top_idx].detach().cpu()
            else:
                pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_labels = torch.zeros((0,), dtype=torch.long)
                pred_scores = torch.zeros((0,), dtype=torch.float32)

            for cls, score in zip(pred_labels.tolist(), pred_scores.tolist()):
                per_class_scores[str(int(cls))].append(float(score))

            gt_boxes = gt.get("boxes", torch.zeros((0, 4))).detach().cpu()
            gt_labels = gt.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu()

            # ── Choose saliency method ──
            sal: np.ndarray | None = None
            used_method = xai_method

            if xai_method == "activation" and activation_saliency is not None:
                # Use the pre-computed activation saliency (fast path).
                sal = activation_saliency[idx]
            elif xai_method == "occlusion":
                # Per-box saliency via detection-conditioned occlusion sensitivity.
                pred_boxes_cpu = pred_boxes.detach().cpu()
                pred_labels_cpu = pred_labels.detach().cpu()
                max_gt_boxes = int(trainer.config.detection_xai_max_gt_boxes)
                max_pred_boxes = int(trainer.config.detection_xai_max_pred_boxes)
                gt_sel = torch.arange(min(max_gt_boxes, int(gt_boxes.shape[0])))
                pred_sel = torch.arange(min(max_pred_boxes, int(pred_boxes_cpu.shape[0])))
                query_boxes_parts: list[Tensor] = []
                query_labels_parts: list[Tensor] = []
                if gt_sel.numel() > 0:
                    query_boxes_parts.append(gt_boxes[gt_sel])
                    query_labels_parts.append(gt_labels[gt_sel])
                if pred_sel.numel() > 0:
                    query_boxes_parts.append(pred_boxes_cpu[pred_sel])
                    query_labels_parts.append(pred_labels_cpu[pred_sel])

                if query_boxes_parts:
                    q_boxes = torch.cat(query_boxes_parts, dim=0)
                    q_labels = torch.cat(query_labels_parts, dim=0)
                    pred_for_occ = (
                        {
                            k: v.to(device) if isinstance(v, Tensor) else v
                            for k, v in pred.items()
                        }
                        if use_ultra_xai
                        else pred
                    )
                    sal_maps, _baseline_scores = compute_detection_box_saliency_occlusion(
                        model=model_impl,
                        image=images[idx],
                        query_boxes=q_boxes.to(device),
                        query_labels=q_labels.to(device),
                        baseline_pred=pred_for_occ,
                        predict_chw=ultra_predict_chw,
                        device=device,
                        grid_size=int(trainer.config.detection_xai_grid_size),
                        iou_threshold=0.3,
                    )
                    if sal_maps:
                        sal = np.mean(np.stack(sal_maps, axis=0), axis=0).astype(np.float32)

            img_h, img_w = images.shape[2], images.shape[3]

            def _box_saliency_stats(
                boxes_tensor: Tensor,
                labels_tensor: Tensor,
                scores_tensor: Tensor | None = None,
                saliency_map: np.ndarray | None = sal,
                _img_h: int = img_h,
                _img_w: int = img_w,
            ) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                if saliency_map is None or boxes_tensor.numel() == 0:
                    return out
                for bi in range(int(boxes_tensor.shape[0])):
                    bx1 = int(max(0, min(_img_w - 1, float(boxes_tensor[bi, 0].item()))))
                    by1 = int(max(0, min(_img_h - 1, float(boxes_tensor[bi, 1].item()))))
                    bx2 = int(max(0, min(_img_w, float(boxes_tensor[bi, 2].item()))))
                    by2 = int(max(0, min(_img_h, float(boxes_tensor[bi, 3].item()))))
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    patch = saliency_map[by1:by2, bx1:bx2]
                    mean_sal = float(np.mean(patch)) if patch.size > 0 else 0.0
                    max_sal = float(np.max(patch)) if patch.size > 0 else 0.0
                    row: dict[str, Any] = {
                        "box": [bx1, by1, bx2, by2],
                        "label": int(labels_tensor[bi].item()) if bi < int(labels_tensor.numel()) else -1,
                        "saliency_mean": mean_sal,
                        "saliency_max": max_sal,
                    }
                    if scores_tensor is not None and bi < int(scores_tensor.numel()):
                        row["score"] = float(scores_tensor[bi].item())
                    out.append(row)
                return out

            details_row = {
                "image_size": [int(img_h), int(img_w)],
                "xai_method": used_method,
                "gt": _box_saliency_stats(
                    gt.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                    gt.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                ),
                "pred": _box_saliency_stats(
                    pred_boxes,
                    pred_labels,
                    pred_scores,
                ),
            }
            trainer._latest_detection_xai_details.append(details_row)

            p_gt, p_sal, p_pred = save_detection_xai_panels(
                image=np_images[idx],
                saliency=sal,
                boxes_gt=gt.get("boxes"),
                labels_gt=gt.get("labels"),
                boxes_pred=pred_boxes,
                labels_pred=pred_labels,
                scores_pred=pred_scores,
                class_names=class_names,
                save_path=save_dir / f"xai_{idx}.png",
                size=trainer.config.report_xai_size,
            )
            paths.extend((p_gt, p_sal, p_pred))

        # Enrich insight text with per-class AP for better detection diagnostics.
        # Prefer cached full-eval predictions when available (from epoch_end_eval).
        from bnnr.detection_metrics import calculate_per_class_ap

        cached_preds = getattr(trainer.model, "last_eval_preds", [])
        cached_targets = getattr(trainer.model, "last_eval_targets", [])
        if cached_preds and cached_targets:
            ap_preds = cached_preds
            ap_targets = cached_targets
        else:
            ap_preds = [
                {
                    "boxes": p.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                    "scores": p.get("scores", torch.zeros((0,))).detach().cpu(),
                    "labels": p.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                }
                for p in preds
            ]
            ap_targets = [
                {
                    "boxes": t.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                    "labels": t.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                }
                for t in targets
            ]

        per_class_ap = calculate_per_class_ap(
            ap_preds, ap_targets, class_names=class_names,
        )

        xai_insights: dict[str, str] = {}
        xai_diagnoses: dict[str, dict[str, Any]] = {}
        for cls_id, scores in per_class_scores.items():
            if not scores:
                continue
            avg_score = float(np.mean(scores))
            support = int(len(scores))
            ap_info = per_class_ap.get(cls_id, {"ap": 0.0, "support": 0})
            ap_val = float(ap_info.get("ap", 0.0))
            cls_name = (
                class_names[int(cls_id)]
                if class_names is not None and int(cls_id) < len(class_names)
                else f"class_{cls_id}"
            )
            xai_insights[cls_id] = (
                f"{cls_name}: AP@0.5={ap_val:.2f}, mean detection confidence={avg_score:.2f}, "
                f"detections={support} ({used_method} XAI overlays available)."
            )
            xai_diagnoses[cls_id] = {
                "severity": "ok" if ap_val >= 0.5 else ("warning" if ap_val >= 0.2 else "critical"),
                "quality_score": ap_val,
                "trend": "stable",
                "short_text": xai_insights[cls_id],
                "confused_with": [],
                "ap_50": ap_val,
                "detections": support,
                "mean_confidence": avg_score,
            }
        return paths, xai_insights, xai_diagnoses

    if not isinstance(trainer.model, XAICapableModel):
        return [], {}, {}

    _probe.initialize_report_probe_samples(trainer)
    if trainer._report_probe_images is None or trainer._report_probe_labels is None:
        return [], {}, {}
    images = trainer._report_probe_images
    labels = trainer._report_probe_labels

    model_impl = trainer.model.get_model()
    device = next(model_impl.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    maps = generate_saliency_maps(
        model=model_impl,
        images=images,
        labels=labels,
        target_layers=trainer.model.get_target_layers(),
        method=trainer.config.xai_method,
    )

    # Lightweight predictions for insight context (no extra forward pass cost
    # when the model is already in eval mode from generate_saliency_maps).
    # For multi-label: XAI maps target one class, so we use dominant (argmax).
    with torch.no_grad():
        logits = model_impl(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    true_labels = labels.detach().cpu().numpy().tolist()

    # Classification XAI does not resolve class names here (handled upstream).
    class_names = None

    # Enriched analysis when confusion data is available
    confusion_matrix: list[list[int]] | None = None
    if confusion and isinstance(confusion.get("matrix"), list):
        confusion_matrix = confusion["matrix"]

    xai_insights, xai_diagnoses, batch_stats = analyze_xai_batch_rich(
        maps,
        true_labels,
        preds,
        class_names=class_names,
        xai_method=trainer.config.xai_method,
        confusion_matrix=confusion_matrix,
        prev_batch_stats=trainer._prev_xai_batch_stats,
        augmentation_name=augmentation_name if augmentation_name != "baseline" else None,
        baseline_batch_stats=trainer._baseline_xai_stats or None,
    )
    # Store batch stats for the next checkpoint's trend comparison
    trainer._prev_xai_batch_stats = batch_stats

    np_images = _img.tensor_batch_to_preview_uint8(images.detach().cpu())

    save_dir = trainer.config.report_dir / "xai" / f"iter_{iteration}_{augmentation_name}"
    paths = save_xai_visualization(
        np_images,
        maps,
        save_dir=save_dir,
        prefix="xai",
        output_size=trainer.config.report_xai_size,
    )
    return paths, xai_insights, xai_diagnoses


def generate_xai_lightweight(trainer, iteration: int, augmentation_name: str, confusion: dict[str, Any] | None = None) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
    """Lightweight XAI probe: analysis only, no PNG saving.

    Returns (insights, diagnoses, batch_stats).  Cost: one forward pass
    on the probe set + saliency computation (~50–100 ms per call).
    """
    if not trainer._xai_enabled or not isinstance(trainer.model, XAICapableModel):
        return {}, {}, {}

    # Detection XAI is handled in PR3; skip for now
    if trainer._is_detection:
        return {}, {}, {}

    _probe.initialize_report_probe_samples(trainer)
    if trainer._report_probe_images is None or trainer._report_probe_labels is None:
        return {}, {}, {}
    images = trainer._report_probe_images
    labels = trainer._report_probe_labels

    model_impl = trainer.model.get_model()
    device = next(model_impl.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    maps = generate_saliency_maps(
        model=model_impl,
        images=images,
        labels=labels,
        target_layers=trainer.model.get_target_layers(),
        method=trainer.config.xai_method,
    )

    # For multi-label: XAI maps target one class, so we use dominant (argmax).
    with torch.no_grad():
        logits = model_impl(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    true_labels = labels.detach().cpu().numpy().tolist()

    confusion_matrix: list[list[int]] | None = None
    if confusion and isinstance(confusion.get("matrix"), list):
        confusion_matrix = confusion["matrix"]

    xai_insights, xai_diagnoses, batch_stats = analyze_xai_batch_rich(
        maps,
        true_labels,
        preds,
        class_names=None,
        xai_method=trainer.config.xai_method,
        confusion_matrix=confusion_matrix,
        prev_batch_stats=trainer._prev_xai_batch_stats,
        augmentation_name=augmentation_name,
        baseline_batch_stats=trainer._baseline_xai_stats or None,
    )
    return xai_insights, xai_diagnoses, batch_stats


def generate_augmentation_previews(trainer, iteration: int, augmentation_name: str, augmentations: list[BaseAugmentation] | None) -> list[tuple[Path, Path]]:
    if trainer._is_detection:
        from bnnr.detection_xai import draw_boxes_on_image

        _probe.initialize_report_probe_samples(trainer)
        if trainer._report_probe_images is None or trainer._report_probe_targets is None:
            return []

        images = trainer._report_probe_images
        targets = copy.deepcopy(trainer._report_probe_targets)
        batch: Any = (images, targets)

        if augmentations:
            preview_augs = [copy.copy(a) for a in augmentations]
            for pa in preview_augs:
                pa.probability = 1.0
            for pa in preview_augs:
                batch = trainer._apply_augmentation_to_batch(batch, pa)

        aug_images, aug_targets = batch
        np_images = _img.tensor_batch_to_preview_uint8(images)
        np_aug_images = _img.tensor_batch_to_preview_uint8(aug_images)
        save_dir = trainer.config.report_dir / "samples" / f"iter_{iteration}_{augmentation_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        pairs: list[tuple[Path, Path]] = []

        for idx in range(np_images.shape[0]):
            original = np_images[idx]
            augmented = np_aug_images[idx]
            if original.shape[-1] == 1:
                original = np.repeat(original, 3, axis=2)
            if augmented.shape[-1] == 1:
                augmented = np.repeat(augmented, 3, axis=2)

            original = draw_boxes_on_image(
                original,
                targets[idx].get("boxes", torch.zeros((0, 4))),
                targets[idx].get("labels", torch.zeros((0,), dtype=torch.long)),
                class_names=trainer.config.detection_class_names,
            )
            augmented = draw_boxes_on_image(
                augmented,
                aug_targets[idx].get("boxes", torch.zeros((0, 4))),
                aug_targets[idx].get("labels", torch.zeros((0,), dtype=torch.long)),
                class_names=trainer.config.detection_class_names,
            )

            original = cv2.resize(
                original,
                (trainer.config.report_preview_size, trainer.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            augmented = cv2.resize(
                augmented,
                (trainer.config.report_preview_size, trainer.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            orig_path = save_dir / f"sample_{idx}_original.png"
            aug_path = save_dir / f"sample_{idx}_augmented.png"
            cv2.imwrite(str(orig_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            pairs.append((orig_path, aug_path))
        return pairs

    _probe.initialize_report_probe_samples(trainer)
    if trainer._report_probe_images is None or trainer._report_probe_labels is None:
        return []
    images = trainer._report_probe_images
    labels = trainer._report_probe_labels
    sample_indices_np = None
    np_images = _img.tensor_batch_to_preview_uint8(images)

    augmented = np_images.copy()
    if augmentations:
        preview_augs = [copy.copy(a) for a in augmentations]
        for pa in preview_augs:
            pa.probability = 1.0
        np_labels = labels.detach().cpu().numpy()
        for pa in preview_augs:
            if hasattr(pa, "apply_batch_with_labels"):
                augmented = pa.apply_batch_with_labels(augmented, np_labels, sample_indices=sample_indices_np)
            else:
                augmented = pa.apply_batch(augmented)

    save_dir = trainer.config.report_dir / "samples" / f"iter_{iteration}_{augmentation_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pairs = []  # list[tuple[Path, Path]] — reassigned from earlier scope
    labels_np = labels.detach().cpu().numpy().squeeze()
    if labels_np.ndim == 0:
        labels_np = labels_np.reshape(1)
    for idx in range(np_images.shape[0]):
        original = np_images[idx]
        aug = augmented[idx]  # type: ignore[assignment]
        class_id = int(labels_np[idx])
        if original.shape[-1] == 1:
            original = np.repeat(original, 3, axis=2)
        if aug.shape[-1] == 1:  # type: ignore[union-attr,attr-defined]
            aug = np.repeat(aug, 3, axis=2)  # type: ignore[assignment]
        # Save larger thumbnails for clearer inspection in HTML report.
        original = cv2.resize(
            original,
            (trainer.config.report_preview_size, trainer.config.report_preview_size),
            interpolation=cv2.INTER_NEAREST,
        )
        aug = cv2.resize(
            aug,
            (trainer.config.report_preview_size, trainer.config.report_preview_size),
            interpolation=cv2.INTER_NEAREST,
        )
        orig_path = save_dir / f"sample_{idx}_class_{class_id}_original.png"
        aug_path = save_dir / f"sample_{idx}_class_{class_id}_augmented.png"
        cv2.imwrite(str(orig_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(aug_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        pairs.append((orig_path, aug_path))
    return pairs


def generate_dual_xai_analysis(trainer) -> dict[str, Any]:
    if not trainer._xai_enabled or not trainer.config.dual_xai_report or not isinstance(trainer.model, XAICapableModel):
        return {}
    # Detection XAI uses different saliency approach; skip dual analysis
    if trainer._is_detection:
        return {}

    first_batch = next(iter(trainer.val_loader))
    if len(first_batch) == 3:
        images, labels, _ = first_batch
    else:
        images, labels = first_batch
    images = images[: trainer.config.xai_samples]
    labels = labels[: trainer.config.xai_samples]

    device = next(trainer.model.get_model().parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    target_layers = trainer.model.get_target_layers()

    opticam_maps = generate_saliency_maps(
        trainer.model.get_model(),
        images,
        labels,
        target_layers,
        method="opticam",
    )
    craft_maps = generate_saliency_maps(
        trainer.model.get_model(),
        images,
        labels,
        target_layers,
        method="craft",
    )
    target_h, target_w = int(images.shape[-2]), int(images.shape[-1])
    opticam_maps = _img.resize_saliency_batch(opticam_maps, target_h, target_w)
    craft_maps = _img.resize_saliency_batch(craft_maps, target_h, target_w)
    np_images = _img.tensor_batch_to_preview_uint8(
        images.detach().cpu(),
        denorm_mean=trainer.config.denormalization_mean,
        denorm_std=trainer.config.denormalization_std,
    )

    run_dir = getattr(trainer.reporter, "run_dir", trainer.config.report_dir)
    opticam_paths = save_xai_visualization(
        np_images,
        opticam_maps,
        save_dir=Path(run_dir) / "assets" / "xai_compare" / "opticam",
        prefix="opticam",
        output_size=trainer.config.report_xai_size,
    )
    craft_paths = save_xai_visualization(
        np_images,
        craft_maps,
        save_dir=Path(run_dir) / "assets" / "xai_compare" / "craft",
        prefix="craft",
        output_size=trainer.config.report_xai_size,
    )

    diff = np.abs(opticam_maps - craft_maps)
    return {
        "enabled": True,
        "opticam_paths": [str(path) for path in opticam_paths],
        "craft_paths": [str(path) for path in craft_paths],
        "mean_saliency_opticam": float(np.mean(opticam_maps)),
        "mean_saliency_craft": float(np.mean(craft_maps)),
        "mean_abs_difference": float(np.mean(diff)),
    }



def precompute_xai_cache(trainer) -> XAICache | None:
    # Detection uses bbox-prior ICD (DetectionICD/DetectionAICD) which
    # doesn't need an XAI cache.  Skip entirely for detection tasks.
    if trainer._is_detection:
        return None

    if not isinstance(trainer.model, XAICapableModel):
        trainer._log("Model is not XAICapableModel. Disabling XAI.")
        trainer._runtime.xai_disabled = True
        return None

    needs_cache = any(isinstance(aug, (ICD, AICD)) for aug in trainer.augmentations)
    if not needs_cache:
        return None

    # Default the cache under the (timestamped) run directory so saliency maps are
    # never silently reused across runs. An explicit config.xai_cache_dir still wins.
    run_dir = getattr(trainer.reporter, "run_dir", None)
    default_cache_dir = (run_dir / "xai_cache") if run_dir is not None else (trainer.config.checkpoint_dir / "xai_cache")
    cache_dir = trainer.config.xai_cache_dir or default_cache_dir
    cache = XAICache(cache_dir)

    # Resolve n_samples: 0 means "cache all", capped by xai_cache_max_samples
    n_samples = trainer.config.xai_cache_samples
    if n_samples <= 0:
        dataset_size = len(trainer.train_loader.dataset)  # type: ignore[arg-type]
        n_samples = min(dataset_size, trainer.config.xai_cache_max_samples)
        trainer._log(
            f"XAI cache auto-sized to {n_samples} samples "
            f"(dataset={dataset_size}, cap={trainer.config.xai_cache_max_samples})"
        )

    written = cache.precompute_cache(
        model=trainer.model.get_model(),
        train_loader=trainer.train_loader,
        target_layers=trainer.model.get_target_layers(),
        n_samples=n_samples,
        method=trainer.config.xai_method,
        force_recompute=trainer.config.xai_cache_force_recompute,
        show_progress=trainer.config.xai_cache_progress and trainer.config.verbose,
    )
    trainer._log(f"Precomputed {written} XAI cache maps")

    evicted = cache.trim_to_max_mb(trainer.config.xai_cache_max_mb)
    if evicted > 0:
        trainer._log(
            f"XAI cache exceeded {trainer.config.xai_cache_max_mb} MB; "
            f"evicted {evicted} oldest maps"
        )

    for aug in trainer.augmentations:
        if isinstance(aug, (ICD, AICD)):
            aug.cache = cache
    return cache



def generate_augmentation_hints(trainer, xai_diagnoses: dict[str, dict[str, Any]], batch_stats: dict[str, list[dict[str, float]]], *, phase: str = "baseline") -> list[str]:
    if not xai_diagnoses and not batch_stats:
        return []

    hints = _callbacks.saliency_recommendations(batch_stats, xai_diagnoses)

    if hints:
        trainer.logger.info(
            "XAI augmentation hints (%s): %s",
            phase,
            " | ".join(hints),
        )
        if trainer.config.verbose:
            for hint in hints:
                print(f"    [XAI hint] {hint}", flush=True)

    return hints


