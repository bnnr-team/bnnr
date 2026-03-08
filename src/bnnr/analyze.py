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
)
from bnnr.analysis.findings import build_findings
from bnnr.analysis.recommendations import build_recommendations as build_recommendations_structured
from bnnr.analysis.schema import REPORT_SCHEMA_VERSION, serialize_for_json
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
    worst_predictions: list[dict[str, Any]] = field(default_factory=list)
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
            "worst_predictions": self.worst_predictions,
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

    def worst_predictions_list(self, n: int = 20) -> list[dict[str, Any]]:
        """Return top-n worst predictions (by loss or confidence)."""
        return self.worst_predictions[:n]

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
) -> AnalysisReport:
    """Run full analysis on a model (adapter) and validation loader.

    Performs: forward pass → metrics + per-class + confusion; optional XAI,
    data quality, failure analysis, patterns, and recommendations.
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

    if xai_enabled and task == "classification" and isinstance(adapter, XAICapableModel):
        _run_xai(adapter, val_loader, config, report, xai_method)

    if task == "classification":
        _run_failure_analysis(
            adapter, val_loader, config, report, max_worst, output_dir, xai_enabled, xai_method
        )
        report.failure_patterns = _build_failure_patterns(report)
        report.recommendations = _build_recommendations(report)
        _build_extended_analysis(report)

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
    if batch_stats:
        from bnnr.xai_analysis import compute_xai_quality_score

        qualities = []
        for cls_id, stats_list in batch_stats.items():
            correct = [
                true_labels[i] == preds[i]
                for i in range(len(true_labels))
                if int(true_labels[i]) == int(cls_id)
            ]
            if stats_list and len(correct) == len(stats_list):
                q, _ = compute_xai_quality_score(stats_list, correct)
                qualities.append(q)
        report.xai_quality_summary = {
            "mean_quality_score": float(np.mean(qualities)) if qualities else 0.0,
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
    """Collect per-sample preds/labels/loss, rank worst, optionally add XAI overlays."""
    out = collect_eval_predictions(adapter, val_loader, config)
    if out is None:
        return
    preds, labels, indices, confidences, losses = out
    wrong = (preds != labels).astype(np.float64)
    sort_key = -wrong * 1e6 - losses
    order = np.argsort(sort_key)
    n = min(max_worst, len(order))
    worst_list: list[dict[str, Any]] = []
    for i in range(n):
        idx = int(order[i])
        worst_list.append({
            "index": int(indices[idx]),
            "true_label": int(labels[idx]),
            "pred_label": int(preds[idx]),
            "confidence": float(confidences[idx]),
            "loss": float(losses[idx]),
        })
    report.worst_predictions = worst_list


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

    key_findings = [f.title for f in findings[:5]] if findings else []
    if not key_findings and acc < 0.9:
        key_findings = [f"Overall accuracy is {acc:.0%}"]

    top_actions = [r.title for r in recs_structured[:5]] if recs_structured else []
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
    report.data_quality_result = result.get("data_quality", result)


def _render_analysis_html(report: AnalysisReport) -> str:
    """Generate HTML report (dashboard-aligned layout and styling)."""
    from bnnr.analysis.html_report import render_analysis_html as _render

    return _render(report)
