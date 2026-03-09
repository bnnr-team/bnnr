"""Structured recommendations linked to findings and evidence."""

from __future__ import annotations

from typing import Any

from bnnr.analysis.schema import Finding, Recommendation


def build_recommendations(
    findings: list[Finding],
    report: Any,
    *,
    max_items: int = 15,
) -> list[Recommendation]:
    """Build prioritized, causal recommendations from findings and report.

    report: object with metrics, xai_quality_summary, failure_patterns (legacy list).
    """
    recs: list[Recommendation] = []
    metrics = getattr(report, "metrics", {}) or {}
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    acc = metrics.get("accuracy")

    # Map finding types to priority (lower = do first)
    priority_by_type = {
        "zero_recall_class": 1,
        "class_collapse_suspected": 1,
        "dominant_prediction_bias": 2,
        "low_overall_accuracy": 2,
        "near_zero_recall": 3,
        "minority_class_suppression": 3,
        "confused_pair": 4,
        "high_confidence_wrong": 4,
        "low_confidence_ambiguity": 5,
        "low_xai_quality": 5,
        "background_focus_suspected": 5,
        "artifact_focus_suspected": 6,
    }

    for i, f in enumerate(findings):
        fid = f"finding_{i}"
        priority = priority_by_type.get(f.finding_type, 10)
        action = f.recommended_action or "Review evidence and consider data or augmentation changes."
        rec = Recommendation(
            title=f.title,
            scope=", ".join(f.class_ids) if f.class_ids else "global",
            why=f.description,
            action=action,
            expected_impact=f"Address {f.finding_type}; may improve accuracy and per-class metrics.",
            confidence=f.confidence,
            priority=priority,
            linked_finding_ids=[fid],
        )
        recs.append(rec)

    # Overall XAI recommendation if no finding covered it
    mean_q = xai_summary.get("mean_quality_score")
    if mean_q is not None and mean_q < 0.5 and not any(f.finding_type == "low_xai_quality" for f in findings):
        recs.append(
            Recommendation(
                title="Improve XAI quality globally",
                scope="global",
                why=f"Mean XAI quality score is {mean_q:.2f}; model may focus on background.",
                action="Run BNNR train with XAI-enabled augmentation search (ICD or AICD).",
                expected_impact="Higher saliency focus on objects; better interpretability.",
                confidence="medium",
                priority=6,
                example_command="bnnr train -c config.yaml --dataset <name> --preset standard",
            )
        )

    # Low accuracy generic
    if acc is not None and acc < 0.7 and not any("accuracy" in r.title.lower() for r in recs):
        recs.append(
            Recommendation(
                title="Improve overall accuracy",
                scope="global",
                why=f"Accuracy {acc:.2f} is below 70%.",
                action="Consider more data, stronger augmentation (e.g. ICD/AICD), or architecture changes.",
                expected_impact="Higher accuracy and more reliable predictions.",
                confidence="medium",
                priority=7,
            )
        )

    recs.sort(key=lambda r: (r.priority, r.title))
    return recs[:max_items]
