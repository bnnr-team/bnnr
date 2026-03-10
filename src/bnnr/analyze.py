"""Standalone model analysis (bnnr analyze): metrics, XAI, data quality, failure analysis.

Zero-friction diagnostics for a trained model without running full BNNR training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from bnnr.adapter import XAICapableModel
from bnnr.analysis.class_diagnostics import (
    build_distribution_summary,
    compute_class_diagnostics,
    compute_global_cohen_kappa,
)
from bnnr.analysis.cross_validation import run_cross_validation_from_predictions
from bnnr.analysis.findings import build_findings
from bnnr.analysis.recommendations import (
    build_recommendations as build_recommendations_structured,
)
from bnnr.analysis.schema import (
    REPORT_SCHEMA_VERSION,
    XAIClassSummary,
    serialize_for_json,
)
from bnnr.evaluation import collect_eval_predictions, run_evaluation
from bnnr.utils import ensure_dir
from bnnr.xai import generate_saliency_maps
from bnnr.xai_analysis import analyze_xai_batch_rich


@dataclass
class AnalysisReport:
    """Result of analyze_model: metrics, per-class details, confusion, findings, recommendations."""

    metrics: dict[str, float] = field(default_factory=dict)
    per_class_accuracy: dict[str, dict[str, float | int]] = field(default_factory=dict)
    confusion: dict[str, Any] = field(default_factory=dict)
    xai_insights: dict[str, str] = field(default_factory=dict)
    xai_diagnoses: dict[str, dict[str, Any]] = field(default_factory=dict)
    xai_quality_summary: dict[str, Any] = field(default_factory=dict)
    data_quality_result: dict[str, Any] = field(default_factory=dict)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    # Extended v0.2 schema
    schema_version: str = "0.2.0"
    executive_summary: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    recommendations_structured: list[dict[str, Any]] = field(default_factory=list)
    class_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    true_distribution: dict[str, int] = field(default_factory=dict)
    pred_distribution: dict[str, int] = field(default_factory=dict)
    distribution_summary: dict[str, Any] = field(default_factory=dict)
    failure_patterns_extended: list[dict[str, Any]] = field(default_factory=list)
    # Optional extended sections (classification-focused)
    xai_quality_per_class: dict[str, Any] = field(default_factory=dict)
    xai_examples_per_class: dict[str, Any] = field(default_factory=dict)
    data_quality_summary: dict[str, Any] = field(default_factory=dict)
    cv_results: dict[str, Any] = field(default_factory=dict)
    # v0.2.1: confusion pair XAI analysis and best/worst per-class examples
    confusion_pair_xai: list[dict[str, Any]] = field(default_factory=list)
    best_worst_examples: dict[str, dict[str, list]] = field(default_factory=dict)
    # Cached predictions for downstream analysis (not serialized)
    _cached_preds: Any = field(default=None, repr=False)
    _cached_labels: Any = field(default=None, repr=False)
    _cached_indices: Any = field(default=None, repr=False)
    _cached_confidences: Any = field(default=None, repr=False)
    _cached_losses: Any = field(default=None, repr=False)

    def save(self, output_dir: Path | str) -> Path:
        """Write analysis_report.json and optional artifact dirs under output_dir."""
        out = Path(output_dir)
        ensure_dir(out)
        payload = {
            "schema_version": self.schema_version,
            "metrics": self.metrics,
            "per_class_accuracy": self.per_class_accuracy,
            "confusion": self.confusion,
            "xai_insights": self.xai_insights,
            "xai_diagnoses": self.xai_diagnoses,
            "xai_quality_summary": self.xai_quality_summary,
            "data_quality": self.data_quality_result,
            "failure_patterns": self.failure_patterns,
            "recommendations": self.recommendations,
            "executive_summary": self.executive_summary,
            "findings": self.findings,
            "recommendations_structured": self.recommendations_structured,
            "class_diagnostics": self.class_diagnostics,
            "true_distribution": self.true_distribution,
            "pred_distribution": self.pred_distribution,
            "distribution_summary": self.distribution_summary,
            "failure_patterns_extended": self.failure_patterns_extended,
            "xai_quality_per_class": self.xai_quality_per_class,
            "xai_examples_per_class": self.xai_examples_per_class,
            "data_quality_summary": self.data_quality_summary,
            "cv_results": self.cv_results,
            "confusion_pair_xai": self.confusion_pair_xai,
            "best_worst_examples": self.best_worst_examples,
        }
        json_path = out / "analysis_report.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path

    def to_html(self, path: Path | str) -> Path:
        """Write a single HTML report file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        html = _render_analysis_html(self)
        p.write_text(html, encoding="utf-8")
        return p

    def failure_patterns_list(self) -> list[dict[str, Any]]:
        """Return detected failure patterns."""
        return self.failure_patterns


def analyze_model(
    adapter: Any,
    val_loader: Any,
    config: Any = None,
    *,
    task: str = "classification",
    device: str = "auto",
    output_dir: Path | str | None = None,
    run_data_quality: bool = True,
    max_worst: int = 20,
    xai_method: str = "opticam",
    xai_enabled: bool = True,
    data_quality_max_samples: int = 5000,
    cv_folds: int = 0,
    xai_samples: int = 500,
) -> AnalysisReport:
    """Run full analysis on a model (adapter) and validation loader.

    Performs: forward pass -> metrics + per-class + confusion; optional XAI,
    data quality, failure analysis, confusion-pair XAI, best/worst per-class,
    and structured recommendations.
    """
    from bnnr.core import BNNRConfig

    if config is None:
        config = BNNRConfig(
            task=task,
            device=device,
            metrics=["accuracy", "f1_macro", "loss"],
        )

    metrics, per_class, confusion, _preds, _labels = run_evaluation(
        adapter,
        val_loader,
        config,
        return_preds_labels=True,
    )
    report = AnalysisReport(
        metrics=metrics,
        per_class_accuracy=per_class,
        confusion=confusion,
    )

    max_probe = max(5, xai_samples // 10) if xai_samples else 5
    if xai_enabled and task == "classification" and isinstance(adapter, XAICapableModel):
        _run_xai(adapter, val_loader, config, report, xai_method, max_probe_per_class=max_probe)

    if task == "classification":
        _run_failure_analysis(
            adapter, val_loader, config, report, max_worst, output_dir, xai_enabled, xai_method
        )
        report.failure_patterns = _build_failure_patterns(report)
        report.recommendations = _build_recommendations(report)
        _build_extended_analysis(report)

        if cv_folds and cv_folds > 1:
            _run_cross_validation(
                adapter=adapter,
                val_loader=val_loader,
                config=config,
                report=report,
                n_folds=cv_folds,
            )

        if (
            xai_enabled
            and isinstance(adapter, XAICapableModel)
            and output_dir is not None
            and report._cached_preds is not None
        ):
            _build_confusion_pair_xai(
                adapter=adapter,
                val_loader=val_loader,
                report=report,
                xai_method=xai_method,
                output_dir=Path(output_dir),
            )
            _build_best_worst_per_class(
                adapter=adapter,
                val_loader=val_loader,
                report=report,
                xai_method=xai_method,
                output_dir=Path(output_dir),
            )

    if run_data_quality:
        _run_data_quality(val_loader, config, report, output_dir, data_quality_max_samples)

    if output_dir is not None:
        report.save(output_dir)
    return report


def _run_xai(
    adapter: Any,
    val_loader: Any,
    config: Any,
    report: AnalysisReport,
    xai_method: str,
    *,
    max_probe_per_class: int = 5,
    max_classes: int = 10,
) -> None:
    """Build probe set from loader, run saliency + analyze_xai_batch_rich, fill report XAI fields."""
    probe_images, probe_labels = _build_probe_set(
        val_loader, max_per_class=max_probe_per_class, max_classes=max_classes
    )
    if probe_images is None or probe_images.shape[0] == 0 or probe_labels is None:
        return
    assert probe_labels is not None  # narrow for type checker
    model = adapter.get_model()
    target_layers = adapter.get_target_layers()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(probe_images.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    true_labels = probe_labels.cpu().numpy().tolist()
    maps = generate_saliency_maps(
        model,
        probe_images.to(device),
        probe_labels.to(device),
        target_layers,
        method=xai_method,
    )
    confusion_matrix = None
    if isinstance(report.confusion.get("matrix"), list):
        confusion_matrix = report.confusion["matrix"]
    insights, diagnoses, batch_stats = analyze_xai_batch_rich(
        maps,
        true_labels,
        pred_labels=preds,
        xai_method=xai_method,
        confusion_matrix=confusion_matrix,
    )
    report.xai_insights = insights
    report.xai_diagnoses = diagnoses
    # Global XAI quality summary (mean over per-class scores) and per-class summaries.
    per_class: dict[str, Any] = {}
    per_class_scores: list[float] = []
    for cls_id_str, diag in diagnoses.items():
        q = diag.get("quality_score")
        if q is None:
            continue
        stats_list = batch_stats.get(cls_id_str, [])
        flags: list[str] = []
        severity = diag.get("severity")
        if severity and severity != "ok":
            flags.append(f"severity:{severity}")
        trend = diag.get("trend")
        if isinstance(trend, str) and trend in {"improving", "declining"}:
            flags.append(f"trend:{trend}")
        if diag.get("confused_with"):
            flags.append("confused_with")
        components = diag.get("quality_breakdown", {}) or {}
        summary = XAIClassSummary(
            class_id=str(cls_id_str),
            mean_quality=float(q),
            sample_count=len(stats_list),
            flags=flags,
            components={k: round(float(v), 4) for k, v in components.items() if isinstance(v, (int, float))},
        )
        per_class[cls_id_str] = serialize_for_json(summary)
        per_class_scores.append(float(q))
    report.xai_quality_per_class = per_class
    if per_class_scores:
        report.xai_quality_summary = {
            "mean_quality_score": float(np.mean(per_class_scores)),
            "total_samples": sum(
                d.get("sample_count", 0) if isinstance(d, dict) else 0
                for d in per_class.values()
            ),
        }


def _run_failure_analysis(
    adapter: Any,
    val_loader: Any,
    config: Any,
    report: AnalysisReport,
    max_worst: int,
    output_dir: Path | str | None,
    xai_enabled: bool,
    xai_method: str,
) -> None:
    """Collect per-sample preds/labels/loss; cache for downstream analysis."""
    out = collect_eval_predictions(adapter, val_loader, config)
    if out is None:
        return
    preds, labels, indices, confidences, losses = out
    report._cached_preds = preds
    report._cached_labels = labels
    report._cached_indices = indices
    report._cached_confidences = confidences
    report._cached_losses = losses

    if xai_enabled and isinstance(adapter, XAICapableModel) and output_dir is not None:
        _build_xai_examples_for_worst_cases(
            adapter=adapter,
            val_loader=val_loader,
            report=report,
            preds=preds,
            labels=labels,
            indices=indices,
            confidences=confidences,
            xai_method=xai_method,
            output_dir=Path(output_dir),
        )


def _build_xai_examples_for_worst_cases(
    *,
    adapter: XAICapableModel,
    val_loader: Any,
    report: AnalysisReport,
    preds: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    confidences: np.ndarray,
    xai_method: str,
    output_dir: Path,
    max_examples_per_class: int = 8,
    max_classes: int = 10,
) -> None:
    """Generate XAI overlays for high-confidence wrong cases per class and attach to report.

    This focuses on classification worst cases and writes overlays under
    ``output_dir / artifacts / xai_examples``.
    """
    try:
        from bnnr.xai import generate_saliency_maps, save_xai_visualization
    except Exception:
        # XAI backend not available; skip example generation.
        return

    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        return

    # Group candidate indices by true class, prioritising high-confidence wrong predictions.
    num_samples = len(preds)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i in range(num_samples):
        true_c = int(labels[i])
        is_wrong = int(preds[i]) != true_c
        if not is_wrong:
            continue
        by_class[true_c].append(i)

    if not by_class:
        return

    classes_sorted = sorted(
        by_class.keys(),
        key=lambda c: -len(by_class[c]),
    )[:max_classes]

    images: list[Tensor] = []
    label_tensors: list[Tensor] = []
    meta: list[dict[str, Any]] = []
    for cls in classes_sorted:
        for idx_in_array in by_class[cls][:max_examples_per_class]:
            ds_index = int(indices[idx_in_array])
            try:
                sample = dataset[ds_index]
            except Exception:
                continue
            if isinstance(sample, (list, tuple)):
                img = sample[0]
                label = sample[1]
            else:
                img = sample
                label = labels[idx_in_array]
            if not isinstance(img, Tensor):
                continue
            if isinstance(label, Tensor):
                lbl_val = int(label.item())
            else:
                lbl_val = int(labels[idx_in_array])
            images.append(img.unsqueeze(0))
            label_tensors.append(torch.tensor(lbl_val).view(1))
            meta.append(
                {
                    "index": ds_index,
                    "true_label": int(labels[idx_in_array]),
                    "pred_label": int(preds[idx_in_array]),
                    "confidence": float(confidences[idx_in_array]),
                }
            )

    if not images:
        return

    images_batch = torch.cat(images, dim=0)  # [B, C, H, W]
    labels_batch = torch.cat(label_tensors, dim=0)

    model = adapter.get_model()
    target_layers = adapter.get_target_layers()
    device = next(model.parameters()).device
    model.eval()
    maps = generate_saliency_maps(
        model,
        images_batch.to(device),
        labels_batch.to(device),
        target_layers,
        method=xai_method,
    )

    # Convert images to uint8 HWC for visualisation.
    imgs_np: list[np.ndarray] = []
    for img in images_batch:
        img_cpu = img.detach().cpu().float()
        # [C, H, W] -> [H, W, C]
        hwc = img_cpu.permute(1, 2, 0).numpy()
        # Normalise to [0, 255]
        mn = float(hwc.min())
        mx = float(hwc.max())
        if mx > mn:
            hwc = (hwc - mn) / (mx - mn)
        hwc = np.clip(hwc * 255.0, 0.0, 255.0).astype(np.uint8)
        imgs_np.append(hwc)
    images_np = np.stack(imgs_np, axis=0)

    save_dir = output_dir / "artifacts" / "xai_examples"
    paths = save_xai_visualization(images_np, maps, save_dir, prefix="xai_example")

    per_class_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info, path in zip(meta, paths):
        cls_id = str(info["true_label"])
        try:
            rel_path = str(path.relative_to(output_dir))
        except ValueError:
            rel_path = str(path)
        ex = {
            "index": info["index"],
            "true_label": info["true_label"],
            "pred_label": info["pred_label"],
            "confidence": info["confidence"],
            "image_path": rel_path,
            "overlay_path": rel_path,
        }
        per_class_examples[cls_id].append(ex)

    report.xai_examples_per_class = {k: v for k, v in per_class_examples.items()}


def _generate_saliency_for_indices(
    adapter: XAICapableModel,
    val_loader: Any,
    sample_indices: list[int],
    xai_method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Generate saliency maps for specific dataset indices.

    Returns (saliency_maps [N,H,W], images_uint8 [N,H,W,3], stats_per_sample).
    """
    from bnnr.xai_analysis import analyze_saliency_map

    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        return np.array([]), np.array([]), []

    images_t: list[Tensor] = []
    labels_t: list[Tensor] = []
    for ds_idx in sample_indices:
        try:
            sample = dataset[ds_idx]
        except Exception:
            continue
        img = sample[0] if isinstance(sample, (list, tuple)) else sample
        lbl = sample[1] if isinstance(sample, (list, tuple)) and len(sample) > 1 else 0
        if not isinstance(img, Tensor):
            continue
        images_t.append(img.unsqueeze(0))
        lbl_val = int(lbl.item()) if isinstance(lbl, Tensor) else int(lbl)
        labels_t.append(torch.tensor(lbl_val).view(1))

    if not images_t:
        return np.array([]), np.array([]), []

    images_batch = torch.cat(images_t, dim=0)
    labels_batch = torch.cat(labels_t, dim=0)

    model = adapter.get_model()
    target_layers = adapter.get_target_layers()
    device = next(model.parameters()).device
    model.eval()

    maps = generate_saliency_maps(
        model, images_batch.to(device), labels_batch.to(device),
        target_layers, method=xai_method,
    )

    imgs_np: list[np.ndarray] = []
    for img in images_batch:
        hwc = img.detach().cpu().float().permute(1, 2, 0).numpy()
        mn, mx = float(hwc.min()), float(hwc.max())
        if mx > mn:
            hwc = (hwc - mn) / (mx - mn)
        imgs_np.append(np.clip(hwc * 255.0, 0.0, 255.0).astype(np.uint8))

    stats_list = [analyze_saliency_map(m) for m in maps]
    return maps, np.stack(imgs_np, axis=0) if imgs_np else np.array([]), stats_list


def _describe_attention_region(stats: dict[str, float]) -> str:
    """Generate a short heuristic description from saliency stats."""
    cx = stats.get("center_x", 0.5)
    cy = stats.get("center_y", 0.5)
    edge = stats.get("edge_ratio", 0.0)
    coverage = stats.get("coverage", 0.0)

    parts: list[str] = []
    if edge > 0.3:
        parts.append(f"border-focused (edge={edge:.0%})")
    elif abs(cx - 0.5) < 0.15 and abs(cy - 0.5) < 0.15:
        parts.append("centered")
    else:
        region = ""
        if cy < 0.35:
            region = "upper"
        elif cy > 0.65:
            region = "lower"
        if cx < 0.35:
            region += " left" if region else "left"
        elif cx > 0.65:
            region += " right" if region else "right"
        if region:
            parts.append(f"{region.strip()} region")

    if coverage < 0.05:
        parts.append(f"tightly focused ({coverage:.0%} coverage)")
    elif coverage > 0.30:
        parts.append(f"diffuse ({coverage:.0%} coverage)")
    else:
        parts.append(f"moderate spread ({coverage:.0%} coverage)")

    return ", ".join(parts) if parts else "no distinctive pattern"


def _build_confusion_pair_xai(
    *,
    adapter: XAICapableModel,
    val_loader: Any,
    report: AnalysisReport,
    xai_method: str,
    output_dir: Path,
    top_n: int = 3,
    max_samples_per_pair: int = 8,
) -> None:
    """Build XAI-powered analysis for top confused class pairs."""
    from bnnr.xai import save_xai_visualization

    preds = report._cached_preds
    labels = report._cached_labels
    indices = report._cached_indices
    if preds is None or labels is None or indices is None:
        return

    confusion = report.confusion
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    if not isinstance(matrix, list) or not labels_list:
        return

    mat = np.asarray(matrix)
    n = mat.shape[0]
    pairs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] > 0:
                pairs.append((i, j, int(mat[i, j])))
    pairs.sort(key=lambda x: -x[2])

    results: list[dict[str, Any]] = []
    for true_idx, pred_idx, count in pairs[:top_n]:
        true_cls = int(labels_list[true_idx])
        pred_cls = int(labels_list[pred_idx])

        confused_mask = (labels == true_cls) & (preds == pred_cls)
        confused_sample_idx = np.where(confused_mask)[0][:max_samples_per_pair]
        confused_ds_indices = [int(indices[i]) for i in confused_sample_idx]

        correct_mask = (labels == true_cls) & (preds == true_cls)
        correct_sample_idx = np.where(correct_mask)[0][:max_samples_per_pair]
        correct_ds_indices = [int(indices[i]) for i in correct_sample_idx]

        pair_data: dict[str, Any] = {
            "class_a": str(true_cls),
            "class_b": str(pred_cls),
            "count": count,
            "sample_overlays": [],
            "heuristic_description": "",
            "mean_overlay_correct_a": "",
            "mean_overlay_confused_ab": "",
            "stats_correct": {},
            "stats_confused": {},
        }

        pair_dir = output_dir / "artifacts" / "confusion_pairs" / f"{true_cls}_to_{pred_cls}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        if confused_ds_indices:
            maps_c, imgs_c, stats_c = _generate_saliency_for_indices(
                adapter, val_loader, confused_ds_indices, xai_method
            )
            if maps_c.size > 0:
                paths = save_xai_visualization(imgs_c, maps_c, pair_dir, prefix="confused")
                for i, path in enumerate(paths[:max_samples_per_pair]):
                    try:
                        rp = str(path.relative_to(output_dir))
                    except ValueError:
                        rp = str(path)
                    pair_data["sample_overlays"].append({
                        "overlay_path": rp,
                        "true_label": true_cls,
                        "pred_label": pred_cls,
                        "type": "confused",
                    })

                mean_map = maps_c.mean(axis=0)
                mean_img = imgs_c.mean(axis=0).astype(np.uint8)
                mean_paths = save_xai_visualization(
                    mean_img[np.newaxis], mean_map[np.newaxis], pair_dir, prefix="mean_confused"
                )
                if mean_paths:
                    try:
                        pair_data["mean_overlay_confused_ab"] = str(mean_paths[0].relative_to(output_dir))
                    except ValueError:
                        pair_data["mean_overlay_confused_ab"] = str(mean_paths[0])

                avg_stats = {}
                if stats_c:
                    for key in stats_c[0]:
                        vals = [s.get(key, 0.0) for s in stats_c if isinstance(s.get(key), (int, float))]
                        if vals:
                            avg_stats[key] = round(float(np.mean(vals)), 4)
                pair_data["stats_confused"] = avg_stats

        if correct_ds_indices:
            maps_ok, imgs_ok, stats_ok = _generate_saliency_for_indices(
                adapter, val_loader, correct_ds_indices, xai_method
            )
            if maps_ok.size > 0:
                mean_map_ok = maps_ok.mean(axis=0)
                mean_img_ok = imgs_ok.mean(axis=0).astype(np.uint8)
                mean_paths_ok = save_xai_visualization(
                    mean_img_ok[np.newaxis], mean_map_ok[np.newaxis], pair_dir, prefix="mean_correct"
                )
                if mean_paths_ok:
                    try:
                        pair_data["mean_overlay_correct_a"] = str(mean_paths_ok[0].relative_to(output_dir))
                    except ValueError:
                        pair_data["mean_overlay_correct_a"] = str(mean_paths_ok[0])

                avg_ok = {}
                if stats_ok:
                    for key in stats_ok[0]:
                        vals = [s.get(key, 0.0) for s in stats_ok if isinstance(s.get(key), (int, float))]
                        if vals:
                            avg_ok[key] = round(float(np.mean(vals)), 4)
                pair_data["stats_correct"] = avg_ok

        sc = pair_data.get("stats_confused", {})
        so = pair_data.get("stats_correct", {})
        if sc and so:
            desc_confused = _describe_attention_region(sc)
            desc_correct = _describe_attention_region(so)
            pair_data["heuristic_description"] = (
                f"When correctly classified as class {true_cls}, attention is {desc_correct}. "
                f"In confused predictions (predicted as {pred_cls}), attention shifts to {desc_confused}."
            )

        results.append(pair_data)

    report.confusion_pair_xai = results


def _build_best_worst_per_class(
    *,
    adapter: XAICapableModel,
    val_loader: Any,
    report: AnalysisReport,
    xai_method: str,
    output_dir: Path,
    n_best: int = 4,
    n_worst: int = 4,
) -> None:
    """Build 4 best + 4 worst examples per class with saliency overlays."""
    from bnnr.xai import save_xai_visualization

    preds = report._cached_preds
    labels = report._cached_labels
    indices = report._cached_indices
    confidences = report._cached_confidences
    if preds is None or labels is None or indices is None or confidences is None:
        return

    classes = sorted(set(int(lbl) for lbl in labels))
    result: dict[str, dict[str, list]] = {}

    for cls in classes:
        cls_mask = labels == cls
        cls_preds = preds[cls_mask]
        cls_conf = confidences[cls_mask]
        cls_indices_arr = indices[cls_mask]
        cls_local = np.arange(cls_mask.sum())

        correct = cls_preds == cls
        wrong = ~correct

        best_local = cls_local[correct][np.argsort(-cls_conf[correct])][:n_best] if correct.any() else []
        worst_local = cls_local[wrong][np.argsort(-cls_conf[wrong])][:n_worst] if wrong.any() else []

        cls_dir = output_dir / "artifacts" / "class_examples" / f"class_{cls}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        entries: dict[str, list] = {"best": [], "worst": []}

        for tag, local_idx_arr in [("best", best_local), ("worst", worst_local)]:
            if len(local_idx_arr) == 0:
                continue
            ds_indices_for_gen = [int(cls_indices_arr[i]) for i in local_idx_arr]
            maps_arr, imgs_arr, stats_arr = _generate_saliency_for_indices(
                adapter, val_loader, ds_indices_for_gen, xai_method
            )
            if maps_arr.size == 0:
                continue
            paths = save_xai_visualization(imgs_arr, maps_arr, cls_dir, prefix=tag)
            for k, path in enumerate(paths):
                try:
                    rp = str(path.relative_to(output_dir))
                except ValueError:
                    rp = str(path)
                li = local_idx_arr[k]
                desc = _describe_attention_region(stats_arr[k]) if k < len(stats_arr) else ""
                entries[tag].append({
                    "overlay_path": rp,
                    "true_label": cls,
                    "pred_label": int(cls_preds[li]),
                    "confidence": float(cls_conf[li]),
                    "description": f"Attention: {desc}" if desc else "",
                })

        result[str(cls)] = entries

    report.best_worst_examples = result


def _run_cross_validation(
    *,
    adapter: Any,
    val_loader: Any,
    config: Any,
    report: AnalysisReport,
    n_folds: int,
) -> None:
    """Run lightweight k-fold CV on cached predictions to populate report.cv_results."""
    from bnnr.evaluation import collect_eval_predictions

    out = collect_eval_predictions(adapter, val_loader, config)
    if out is None:
        return
    preds, labels, _indices, _confidences, _losses = out
    cv = run_cross_validation_from_predictions(preds, labels, n_folds)
    report.cv_results = serialize_for_json(cv)


def _build_failure_patterns(report: AnalysisReport) -> list[dict[str, Any]]:
    """Build list of failure patterns: top confused pairs, low XAI quality classes."""
    patterns: list[dict[str, Any]] = []
    confusion = report.confusion
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    if isinstance(matrix, list) and labels_list:
        n = len(matrix)
        for true_idx in range(n):
            for pred_idx in range(n):
                if true_idx != pred_idx and matrix[true_idx][pred_idx] > 0:
                    patterns.append({
                        "type": "confused_pair",
                        "true_class": str(labels_list[true_idx]),
                        "pred_class": str(labels_list[pred_idx]),
                        "count": matrix[true_idx][pred_idx],
                    })
        patterns.sort(key=lambda x: -x["count"])
        patterns = patterns[:10]
    for cls_id, diag in report.xai_diagnoses.items():
        q = diag.get("quality_score", 1.0)
        if q < 0.5:
            patterns.append({
                "type": "low_xai_quality",
                "class": cls_id,
                "quality_score": round(q, 4),
            })
    return patterns


def _build_recommendations(report: AnalysisReport) -> list[str]:
    """Build improvement recommendations from metrics, XAI, and patterns."""
    recs: list[str] = []
    acc = report.metrics.get("accuracy", 0.0)
    if acc < 0.7:
        recs.append("Overall accuracy is below 70%. Consider more data, augmentation (e.g. ICD/AICD), or architecture changes.")
    for pat in report.failure_patterns:
        if pat.get("type") == "confused_pair" and pat.get("count", 0) >= 5:
            recs.append(
                f"Model often confuses class {pat.get('true_class')} with {pat.get('pred_class')} "
                f"({pat.get('count')} times). Consider targeted augmentation or more samples for these classes."
            )
        if pat.get("type") == "low_xai_quality":
            recs.append(
                f"Class {pat.get('class')} has low XAI quality (model may focus on background). "
                "Consider ICD or AICD to encourage object-focused features."
            )
    mean_q = (report.xai_quality_summary or {}).get("mean_quality_score")
    if mean_q is not None and mean_q < 0.5:
        recs.append("Overall XAI quality is low. Consider running BNNR train with XAI-enabled augmentation search (ICD/AICD).")
    if not recs:
        recs.append("No strong improvement recommendations; model metrics and XAI quality are acceptable.")
    return recs


_CONFIDENCE_PREFIXES = {"high": "", "medium": "[Likely] ", "low": "[Suspected] "}


def _build_executive_summary(
    report: AnalysisReport,
    findings: list[Any],
    recs_structured: list[Any],
) -> dict[str, Any]:
    """Build 30-second summary: health, key findings, top actions."""
    from bnnr.analysis.schema import ExecutiveSummary

    acc = report.metrics.get("accuracy")
    if acc is None:
        acc = 0.0
    mean_q = (report.xai_quality_summary or {}).get("mean_quality_score")
    if mean_q is None:
        mean_q = 1.0

    if acc >= 0.85 and mean_q >= 0.6:
        health_status = "ok"
        health_score = 0.5 + 0.5 * (acc * 0.6 + mean_q * 0.4)
        severity = "low"
    elif acc >= 0.7 and mean_q >= 0.4:
        health_status = "warning"
        health_score = 0.3 + 0.4 * (acc * 0.5 + mean_q * 0.5)
        severity = "medium"
    else:
        health_status = "critical"
        health_score = max(0.0, 0.2 * acc + 0.2 * mean_q)
        severity = "high" if acc < 0.5 else "medium"

    health_score = min(1.0, health_score)

    key_findings: list[str] = []
    for f in findings[:5]:
        prefix = _CONFIDENCE_PREFIXES.get(f.confidence, "")
        key_findings.append(f"{prefix}{f.title}")
    if not key_findings:
        key_findings = [f"Overall accuracy is {acc:.0%}"]

    top_actions = [r.title for r in recs_structured[:3]] if recs_structured else []
    if not top_actions and report.recommendations:
        top_actions = report.recommendations[:3]

    critical_classes = []
    for d in getattr(report, "class_diagnostics", []) or []:
        if isinstance(d, dict) and d.get("severity") == "critical":
            critical_classes.append(d.get("class_id", ""))
        elif hasattr(d, "severity") and d.severity == "critical":
            critical_classes.append(getattr(d, "class_id", ""))

    summary = ExecutiveSummary(
        health_status=health_status,
        health_score=round(health_score, 2),
        key_findings=key_findings,
        top_actions=top_actions,
        critical_classes=critical_classes[:5],
        severity=severity,
    )
    return cast(dict[str, Any], serialize_for_json(summary))


def _build_extended_analysis(report: AnalysisReport) -> None:
    """Fill executive summary, findings, structured recommendations, class diagnostics."""
    report.schema_version = REPORT_SCHEMA_VERSION

    global_kappa = compute_global_cohen_kappa(report.confusion)
    report.metrics["cohen_kappa"] = round(global_kappa, 4)

    class_diag, true_dist, pred_dist = compute_class_diagnostics(report.confusion)
    report.class_diagnostics = [serialize_for_json(d) for d in class_diag]
    report.true_distribution = true_dist
    report.pred_distribution = pred_dist
    report.distribution_summary = build_distribution_summary(true_dist, pred_dist)

    findings, patterns_ext = build_findings(
        report, class_diag, true_dist, pred_dist
    )
    report.findings = [serialize_for_json(f) for f in findings]
    report.failure_patterns_extended = [serialize_for_json(p) for p in patterns_ext]

    recs_structured = build_recommendations_structured(findings, report, max_items=15)
    report.recommendations_structured = [serialize_for_json(r) for r in recs_structured]

    report.executive_summary = _build_executive_summary(
        report, findings, recs_structured
    )


def _build_probe_set(
    loader: Any,
    max_per_class: int = 5,
    max_classes: int = 10,
) -> tuple[Tensor | None, Tensor | None]:
    """Collect up to max_per_class images per class (max max_classes). Returns (images, labels)."""
    by_class: dict[int, list[tuple[Tensor, Tensor]]] = defaultdict(list)
    for batch in loader:
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch[0], batch[1]
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        for i in range(images.shape[0]):
            lbl = int(labels[i].item())
            if lbl not in by_class or len(by_class[lbl]) < max_per_class:
                by_class[lbl].append((images[i : i + 1], labels[i : i + 1]))
        if len(by_class) >= max_classes:
            break
    classes = sorted(by_class.keys())[:max_classes]
    if not classes:
        return None, None
    imgs, labs = [], []
    for c in classes:
        for img, lab in by_class[c][:max_per_class]:
            imgs.append(img)
            labs.append(lab)
    if not imgs:
        return None, None
    return torch.cat(imgs, dim=0), torch.cat(labs, dim=0)


def _run_data_quality(
    loader: Any,
    config: Any,
    report: AnalysisReport,
    output_dir: Path | str | None,
    max_samples: int,
) -> None:
    from bnnr.data_quality import run_data_quality_analysis

    task = getattr(config, "task", "classification")
    is_detection = task == "detection"
    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir) / "artifacts" / "data_quality"
        save_dir.mkdir(parents=True, exist_ok=True)
    result = run_data_quality_analysis(
        loader,
        max_samples=max_samples,
        is_detection=is_detection,
        save_dir=save_dir,
    )
    data_quality = result.get("data_quality", result)
    report.data_quality_result = data_quality
    if isinstance(data_quality, dict):
        report.data_quality_summary = data_quality

    if save_dir is not None and isinstance(data_quality, dict):
        _embed_thumbnails_as_base64(data_quality, save_dir)
        report.data_quality_summary = data_quality


def _embed_thumbnails_as_base64(dq: dict[str, Any], save_dir: Path) -> None:
    """Read saved thumbnail PNGs from duplicates/ and flagged/ dirs, embed as base64."""
    import base64

    dup_thumbs: list[dict[str, str]] = []
    flagged_thumbs: list[dict[str, str]] = []

    dup_dir = save_dir / "duplicates"
    if dup_dir.exists():
        for group_dir in sorted(dup_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_imgs: list[dict[str, str]] = []
            for png_file in sorted(group_dir.glob("*.png"))[:6]:
                try:
                    data = png_file.read_bytes()
                    b64 = base64.b64encode(data).decode("ascii")
                    group_imgs.append({
                        "filename": png_file.name,
                        "base64": b64,
                        "group": group_dir.name,
                    })
                except Exception:
                    continue
            dup_thumbs.extend(group_imgs)

    flagged_dir = save_dir / "flagged"
    if flagged_dir.exists():
        for png_file in sorted(flagged_dir.glob("*.png"))[:20]:
            try:
                data = png_file.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                flagged_thumbs.append({"filename": png_file.name, "base64": b64})
            except Exception:
                continue

    if dup_thumbs:
        dq["duplicate_thumbnails"] = dup_thumbs
    if flagged_thumbs:
        dq["flagged_thumbnails"] = flagged_thumbs


def _render_analysis_html(report: AnalysisReport) -> str:
    """Generate HTML report (dashboard-aligned layout and styling)."""
    from bnnr.analysis.html_report import render_analysis_html as _render

    return _render(report)
