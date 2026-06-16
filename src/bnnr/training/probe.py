"""Probe sample selection for report artifacts and dashboard previews."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from bnnr.adapter import XAICapableModel
from bnnr.training.checkpoint import _is_ultralytics_tasks_backbone
from bnnr.utils import lazy_cv2 as cv2

if TYPE_CHECKING:
    from bnnr.trainer import BNNRTrainer


def probe_labels_from_tensor(labels: Tensor | None) -> list[int]:
    if labels is None:
        return []
    return [int(v) for v in labels.detach().cpu().numpy().tolist()]


def probe_sample_ids_from_list(sample_ids: list[str]) -> list[str]:
    return list(sample_ids)


def initialize_report_probe_samples(trainer: BNNRTrainer) -> None:
    """Select per-class probe images from the validation loader for reporting."""
    if trainer._report_probe_images is not None and trainer._report_probe_labels is not None:
        return

    images_by_class: dict[int, list[Tensor]] = defaultdict(list)
    targets_by_class: dict[int, list[dict[str, Tensor] | None]] = defaultdict(list)
    indices_by_class: dict[int, list[int | None]] = defaultdict(list)
    for raw_batch in trainer.val_loader:
        if trainer._is_detection:
            if len(raw_batch) == 3:
                images, targets, sample_indices = raw_batch
            else:
                images, targets = raw_batch
                sample_indices = None
            for idx in range(images.shape[0]):
                target = targets[idx]
                if len(target.get("labels", [])) == 0:
                    continue
                unique_classes = set(int(lbl.item()) for lbl in target["labels"])
                for class_id in unique_classes:
                    images_by_class[class_id].append(images[idx].detach().cpu())
                    targets_by_class[class_id].append({
                        "boxes": target["boxes"].detach().cpu().clone(),
                        "labels": target["labels"].detach().cpu().clone(),
                    })
                    if sample_indices is not None:
                        current_idx = sample_indices[idx]
                        if isinstance(current_idx, Tensor):
                            indices_by_class[class_id].append(int(current_idx.item()))
                        else:
                            indices_by_class[class_id].append(int(current_idx))
                    else:
                        indices_by_class[class_id].append(None)
            continue

        if len(raw_batch) == 3:
            images, labels, sample_indices = raw_batch
        else:
            images, labels = raw_batch
            sample_indices = None

        if trainer._is_multilabel:
            for idx in range(images.shape[0]):
                label_vec = labels[idx]
                active_classes = [
                    cls_idx
                    for cls_idx in range(label_vec.shape[0])
                    if int(label_vec[cls_idx]) == 1
                ]
                if not active_classes:
                    active_classes = [int(label_vec.argmax().item())]
                for class_id in active_classes:
                    images_by_class[class_id].append(images[idx].detach().cpu())
                    if sample_indices is not None:
                        si = sample_indices[idx]
                        indices_by_class[class_id].append(
                            int(si.item()) if isinstance(si, Tensor) else int(si)
                        )
                    else:
                        indices_by_class[class_id].append(None)
        else:
            for idx in range(images.shape[0]):
                lbl = labels[idx]
                if lbl.ndim >= 1:
                    lbl = lbl.squeeze()
                class_id = int(lbl.item())
                images_by_class[class_id].append(images[idx].detach().cpu())
                if sample_indices is not None:
                    indices_by_class[class_id].append(int(sample_indices[idx].item()))
                else:
                    indices_by_class[class_id].append(None)

    rnd = np.random.default_rng(trainer.config.seed)
    selected_images: list[Tensor] = []
    selected_labels: list[int] = []
    selected_targets: list[dict[str, Tensor]] = []
    selected_sample_ids: list[str] = []
    classes = sorted(images_by_class.keys())[: trainer.config.report_probe_max_classes]
    per_class = trainer.config.report_probe_images_per_class
    for class_id in classes:
        class_images = images_by_class[class_id]
        class_indices = indices_by_class[class_id]
        if not class_images:
            continue
        sample_count = min(per_class, len(class_images))
        chosen_indices = rnd.choice(len(class_images), size=sample_count, replace=False)
        for probe_rank, index in enumerate(chosen_indices):
            real_index = int(index)
            selected_images.append(class_images[real_index])
            selected_labels.append(class_id)
            if trainer._is_detection:
                target_entry = targets_by_class[class_id][real_index]
                if target_entry is not None:
                    selected_targets.append(target_entry)
            sample_idx = class_indices[real_index]
            if sample_idx is None:
                selected_sample_ids.append(f"class_{class_id}_probe_{probe_rank}")
            else:
                selected_sample_ids.append(f"sample_{sample_idx}")

    if not selected_images:
        return
    trainer._report_probe_images = torch.stack(selected_images)
    trainer._report_probe_labels = torch.as_tensor(selected_labels, dtype=torch.long)
    trainer._report_probe_targets = selected_targets if trainer._is_detection else None
    trainer._report_probe_sample_ids = selected_sample_ids
    log_probe_set = getattr(trainer.reporter, "log_probe_set", None)
    if callable(log_probe_set):
        probes = [
            {"sample_id": sample_id, "class_id": label, "index": idx}
            for idx, (sample_id, label) in enumerate(zip(selected_sample_ids, selected_labels))
        ]
        log_probe_set(probes)

def to_artifact_reference(path: Path, run_dir: Path | None) -> str:
    if run_dir is not None:
        try:
            return str(path.resolve().relative_to(Path(run_dir).resolve())).replace("\\", "/")
        except ValueError:
            pass
    text = str(path).replace("\\", "/")
    marker = "/artifacts/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        return f"artifacts/{suffix}"
    return text

def emit_probe_prediction_snapshots(trainer,
    *,
    iteration: int,
    epoch: int,
    branch: str,
    preview_pairs: list[tuple[Path, Path]],
    xai_paths: list[Path],
    ) -> None:
    log_prediction = getattr(trainer.reporter, "log_sample_prediction", None)
    if not callable(log_prediction):
        return
    if trainer._is_detection:
        model_getter = getattr(trainer.model, "get_model", None)
        if not callable(model_getter):
            return
        initialize_report_probe_samples(trainer)
        if trainer._report_probe_images is None or trainer._report_probe_targets is None:
            return
        model = model_getter()
        device = next(model.parameters()).device
        images = trainer._report_probe_images.to(device)
        predict_ultra = getattr(trainer.model, "predict_detection_dicts", None)
        if _is_ultralytics_tasks_backbone(model) and not callable(predict_ultra):
            trainer._log(
                "Probe prediction snapshots skipped: Ultralytics backbone without "
                "UltralyticsDetectionAdapter.predict_detection_dicts."
            )
            return
        with torch.no_grad():
            if callable(predict_ultra):
                preds = predict_ultra(images)
            else:
                preds = model([img for img in images])

        sample_ids = trainer._probe_sample_ids()
        reporter_run_dir = getattr(trainer.reporter, "run_dir", None)
        fallback_originals = ensure_fast_probe_originals(trainer,
            iteration=iteration,
            branch=branch,
            epoch=epoch,
            run_dir=reporter_run_dir,
        )
        normalized_preview_pairs = preview_pairs
        normalized_xai_paths = xai_paths
        checkpoints = getattr(trainer.reporter, "_checkpoints", None)
        if isinstance(checkpoints, list) and checkpoints:
            latest = checkpoints[-1]
            if getattr(latest, "iteration", None) == iteration and getattr(latest, "augmentation", None) == branch:
                normalized_preview_pairs = list(getattr(latest, "preview_pairs", preview_pairs))
                normalized_xai_paths = list(getattr(latest, "xai_paths", xai_paths))

        n_probes = len(sample_ids)
        xai_triplet_mode = (
            n_probes > 0
            and len(normalized_xai_paths) == 3 * n_probes
        )
        for idx, sample_id in enumerate(sample_ids):
            gt = trainer._report_probe_targets[idx]
            pred = preds[idx]
            pred_scores = pred.get("scores", torch.zeros(0))
            pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
            if pred_scores.numel() > 0:
                top = int(torch.argmax(pred_scores).item())
                predicted_class = int(pred_labels[top].item())
                confidence = float(pred_scores[top].item())
            else:
                predicted_class = -1
                confidence = 0.0

            if gt.get("labels") is not None and int(gt["labels"].numel()) > 0:
                true_class = int(gt["labels"][0].item())
            else:
                true_class = -1

            original_artifact = None
            augmented_artifact = None
            xai_artifact = None
            xai_gt_artifact = None
            xai_saliency_artifact = None
            xai_pred_artifact = None
            if idx < len(normalized_preview_pairs):
                original_artifact = to_artifact_reference(normalized_preview_pairs[idx][0], reporter_run_dir)
                augmented_artifact = to_artifact_reference(normalized_preview_pairs[idx][1], reporter_run_dir)
            elif idx < len(fallback_originals):
                # Keep Samples tab usable immediately after epoch_end:
                # if full previews are not ready yet, expose a fast original.
                original_artifact = to_artifact_reference(fallback_originals[idx], reporter_run_dir)
            if xai_triplet_mode:
                base = idx * 3
                xai_gt_artifact = to_artifact_reference(normalized_xai_paths[base], reporter_run_dir)
                xai_saliency_artifact = to_artifact_reference(
                    normalized_xai_paths[base + 1], reporter_run_dir,
                )
                xai_pred_artifact = to_artifact_reference(
                    normalized_xai_paths[base + 2], reporter_run_dir,
                )
                xai_artifact = xai_saliency_artifact
            elif idx < len(normalized_xai_paths):
                xai_artifact = to_artifact_reference(normalized_xai_paths[idx], reporter_run_dir)

            detection_details: dict[str, Any] = {}
            if idx < len(trainer._latest_detection_xai_details):
                detection_details = trainer._latest_detection_xai_details[idx]

            log_prediction(
                sample_id=sample_id,
                iteration=iteration,
                epoch=epoch,
                branch=branch,
                true_class=true_class,
                predicted_class=predicted_class,
                confidence=confidence,
                loss_local=None,
                original_artifact=original_artifact,
                augmented_artifact=augmented_artifact,
                xai_artifact=xai_artifact,
                xai_gt_artifact=xai_gt_artifact,
                xai_saliency_artifact=xai_saliency_artifact,
                xai_pred_artifact=xai_pred_artifact,
                detection_details=detection_details,
            )
        return

    if not isinstance(trainer.model, XAICapableModel):
        return
    initialize_report_probe_samples(trainer)
    if trainer._report_probe_images is None or trainer._report_probe_labels is None:
        return
    model = trainer.model.get_model()
    device = next(model.parameters()).device
    images = trainer._report_probe_images.to(device)
    labels = trainer._report_probe_labels.to(device)
    with torch.no_grad():
        logits = model(images)
        if trainer._is_multilabel:
            # Multi-label: use sigmoid + threshold; report dominant class
            probs = torch.sigmoid(logits)
            conf, pred = torch.max(probs, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
    sample_ids = trainer._probe_sample_ids()
    true_labels = labels.detach().cpu().numpy().tolist()
    pred_labels = pred.detach().cpu().numpy().tolist()
    conf_scores = conf.detach().cpu().numpy().tolist()
    reporter_run_dir = getattr(trainer.reporter, "run_dir", None)
    normalized_preview_pairs = preview_pairs
    normalized_xai_paths = xai_paths
    checkpoints = getattr(trainer.reporter, "_checkpoints", None)
    if isinstance(checkpoints, list) and checkpoints:
        latest = checkpoints[-1]
        if getattr(latest, "iteration", None) == iteration and getattr(latest, "augmentation", None) == branch:
            normalized_preview_pairs = list(getattr(latest, "preview_pairs", preview_pairs))
            normalized_xai_paths = list(getattr(latest, "xai_paths", xai_paths))
    for idx, sample_id in enumerate(sample_ids):
        original_artifact = None
        augmented_artifact = None
        xai_artifact = None
        if idx < len(normalized_preview_pairs):
            original_artifact = to_artifact_reference(normalized_preview_pairs[idx][0], reporter_run_dir)
            augmented_artifact = to_artifact_reference(normalized_preview_pairs[idx][1], reporter_run_dir)
        if idx < len(normalized_xai_paths):
            xai_artifact = to_artifact_reference(normalized_xai_paths[idx], reporter_run_dir)
        log_prediction(
            sample_id=sample_id,
            iteration=iteration,
            epoch=epoch,
            branch=branch,
            true_class=int(true_labels[idx]),
            predicted_class=int(pred_labels[idx]),
            confidence=float(conf_scores[idx]),
            loss_local=None,
            original_artifact=original_artifact,
            augmented_artifact=augmented_artifact,
            xai_artifact=xai_artifact,
        )



def ensure_fast_probe_originals(trainer: BNNRTrainer,
    *,
    iteration: int,
    branch: str,
    epoch: int,
    run_dir: Path | None,
    ) -> list[Path]:
    """Create lightweight original probe previews for immediate dashboard updates."""
    key = (iteration, branch, epoch)
    cached = trainer._fast_probe_originals.get(key)
    if cached:
        return cached
    if trainer._report_probe_images is None:
        return []

    save_root = (
        Path(run_dir) / "artifacts"
        if run_dir is not None
        else trainer.config.report_dir / "artifacts"
    )
    save_dir = save_root / "samples" / f"iter_{iteration}_{branch}" / f"epoch_{epoch}" / "fast_originals"
    save_dir.mkdir(parents=True, exist_ok=True)

    np_images = trainer._tensor_batch_to_preview_uint8(trainer._report_probe_images)
    paths: list[Path] = []
    if trainer._is_detection and trainer._report_probe_targets is not None:
        from bnnr.detection_xai import draw_boxes_on_image

        for idx in range(np_images.shape[0]):
            original = np_images[idx]
            if original.shape[-1] == 1:
                original = np.repeat(original, 3, axis=2)
            target = trainer._report_probe_targets[idx]
            overlay = draw_boxes_on_image(
                original,
                target.get("boxes", torch.zeros((0, 4))),
                target.get("labels", torch.zeros((0,), dtype=torch.long)),
                class_names=trainer.config.detection_class_names,
            )
            overlay = cv2.resize(
                overlay,
                (trainer.config.report_preview_size, trainer.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            out_path = save_dir / f"sample_{idx}_original.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            paths.append(out_path)
    else:
        for idx in range(np_images.shape[0]):
            original = np_images[idx]
            if original.shape[-1] == 1:
                original = np.repeat(original, 3, axis=2)
            original = cv2.resize(
                original,
                (trainer.config.report_preview_size, trainer.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            out_path = save_dir / f"sample_{idx}_original.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            paths.append(out_path)

    trainer._fast_probe_originals[key] = paths
    return paths

