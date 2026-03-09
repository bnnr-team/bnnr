"""Structured, literature-backed recommendations grouped by finding type.

Each recommendation aggregates multiple findings of the same type so the user
sees 3-5 actionable items instead of one card per affected class.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bnnr.analysis.schema import Finding, Recommendation

_LITERATURE: dict[str, dict[str, str]] = {
    "zero_recall_class": {
        "action": (
            "Apply class-balanced sampling or focal loss "
            "(Lin et al., ICCV 2017) to force the model to learn underrepresented classes. "
            "BNNR's ICD/AICD augmentation search can automatically find transforms "
            "that improve per-class saliency."
        ),
        "impact": "Focal loss typically recovers 2-5 pp recall on tail classes.",
        "reference": "Lin et al., Focal Loss for Dense Object Detection, ICCV 2017",
    },
    "class_collapse_suspected": {
        "action": (
            "Use deferred re-weighting or class-balanced sampling "
            "(Kang et al., ICLR 2020). "
            "Verify training data is not severely label-noisy for the dominant class."
        ),
        "impact": "Decoupled training recovers 3-8 pp on minority classes.",
        "reference": "Kang et al., Decoupling Representation and Classifier, ICLR 2020",
    },
    "dominant_prediction_bias": {
        "action": (
            "Check XAI overlays for spurious correlations (background, watermarks). "
            "Post-hoc temperature scaling (Guo et al., ICML 2017) "
            "reduces overconfident predictions without retraining."
        ),
        "impact": "Temperature scaling reduces ECE by 50-80%.",
        "reference": "Guo et al., On Calibration of Modern Neural Networks, ICML 2017",
    },
    "low_overall_accuracy": {
        "action": (
            "Review data quality first — mislabeled samples degrade accuracy "
            "(Northcutt et al., JAIR 2021). "
            "Then consider stronger augmentation or a larger backbone."
        ),
        "impact": "Cleaning 1-5% noisy labels improves accuracy by 1-3 pp.",
        "reference": "Northcutt et al., Confident Learning, JAIR 2021",
    },
    "near_zero_recall": {
        "action": (
            "Add targeted augmentation or oversample affected classes. "
            "Cost-sensitive learning with inverse-frequency weights "
            "(Buda et al., Neural Networks 2018) is a reliable first step."
        ),
        "impact": "Class-weighting typically recovers 5-15 pp recall on rare classes.",
        "reference": "Buda et al., A Systematic Study of the Class Imbalance Problem, 2018",
    },
    "minority_class_suppression": {
        "action": (
            "Apply class-balanced batch sampling and strong augmentation "
            "(RandAugment; Cubuk et al., 2020) disproportionately on minority classes."
        ),
        "impact": "Balanced sampling + augmentation closes the recall gap by 10-20 pp.",
        "reference": "Cubuk et al., RandAugment, NeurIPS 2020",
    },
    "confused_pair": {
        "action": (
            "Inspect XAI overlays for both classes. "
            "Pair-specific augmentation or metric learning (ArcFace; Deng et al., CVPR 2019) "
            "can increase inter-class separation."
        ),
        "impact": "Targeted pair augmentation reduces confusion by 30-60%.",
        "reference": "Deng et al., ArcFace, CVPR 2019",
    },
    "high_confidence_wrong": {
        "action": (
            "Apply temperature scaling (Guo et al., ICML 2017) or label smoothing "
            "(Muller et al., NeurIPS 2019) to reduce overconfidence."
        ),
        "impact": "Label smoothing reduces overconfident errors by 15-30%.",
        "reference": "Muller et al., When Does Label Smoothing Help?, NeurIPS 2019",
    },
    "low_confidence_ambiguity": {
        "action": (
            "Review these samples manually — they often reveal labeling errors or "
            "class boundary issues. Mixup training (Zhang et al., ICLR 2018) "
            "improves calibration on borderline cases."
        ),
        "impact": "Mixup improves calibration and reduces error rates by 1-2 pp.",
        "reference": "Zhang et al., Mixup, ICLR 2018",
    },
    "low_xai_quality": {
        "action": (
            "Run BNNR train with ICD/AICD augmentation search to encourage "
            "object-focused features."
        ),
        "impact": "ICD/AICD improves XAI quality by 0.1-0.3 on 0-1 scale.",
        "reference": "Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018",
    },
    "background_focus_suspected": {
        "action": (
            "Apply random cropping, CutOut (DeVries & Taylor, 2017), "
            "or BNNR's ICD/AICD which explicitly penalizes background-focused saliency."
        ),
        "impact": "CutOut + saliency optimization reduces background reliance by 40-60%.",
        "reference": "DeVries & Taylor, Improved Regularization with Cutout, 2017",
    },
    "artifact_focus_suspected": {
        "action": (
            "Apply GridMask (Chen et al., 2020) or center-crop augmentation "
            "to force the model to focus on a single discriminative region."
        ),
        "impact": "GridMask + standard augmentation improves focus score by 15-30%.",
        "reference": "Chen et al., GridMask Data Augmentation, 2020",
    },
}

_PRIORITY: dict[str, int] = {
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

_GROUP_TITLES: dict[str, str] = {
    "zero_recall_class": "Fix zero-recall classes",
    "near_zero_recall": "Recover near-zero recall classes",
    "class_collapse_suspected": "Address class collapse",
    "dominant_prediction_bias": "Reduce prediction bias",
    "low_overall_accuracy": "Improve overall accuracy",
    "minority_class_suppression": "Address minority class suppression",
    "confused_pair": "Reduce top class confusions",
    "high_confidence_wrong": "Fix high-confidence errors",
    "low_confidence_ambiguity": "Review ambiguous predictions",
    "low_xai_quality": "Improve saliency quality",
    "background_focus_suspected": "Fix background attention",
    "artifact_focus_suspected": "Fix diffuse attention patterns",
}

_CONFIDENCE_LABEL: dict[str, str] = {
    "high": "Observed",
    "medium": "Likely",
    "low": "Suspected",
}


def build_recommendations(
    findings: list[Finding],
    report: Any,
    *,
    max_items: int = 5,
) -> list[Recommendation]:
    """Build prioritized, grouped recommendations from findings.

    Multiple findings of the same type are merged into a single recommendation
    listing all affected classes, so the user sees 3-5 actionable items.
    """
    metrics = getattr(report, "metrics", {}) or {}
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    acc = metrics.get("accuracy")

    groups: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        groups[f.finding_type].append(f)

    recs: list[Recommendation] = []
    for ftype, group in groups.items():
        lit = _LITERATURE.get(ftype, {})
        priority = _PRIORITY.get(ftype, 10)
        title = _GROUP_TITLES.get(ftype, ftype.replace("_", " ").title())

        all_classes: list[str] = []
        for f in group:
            all_classes.extend(f.class_ids)
        unique_classes = list(dict.fromkeys(all_classes))

        if unique_classes:
            scope = ", ".join(unique_classes[:6])
            if len(unique_classes) > 6:
                scope += f" (+{len(unique_classes) - 6} more)"
            title = f"{title}: {scope}"
        else:
            scope = "global"

        best_confidence = "low"
        for f in group:
            if f.confidence == "high":
                best_confidence = "high"
                break
            if f.confidence == "medium" and best_confidence == "low":
                best_confidence = "medium"

        confidence_label = _CONFIDENCE_LABEL.get(best_confidence, "")

        descriptions = []
        for f in group[:3]:
            descriptions.append(f.description)
        if len(group) > 3:
            descriptions.append(f"...and {len(group) - 3} more similar findings.")
        why = " ".join(descriptions)

        action = lit.get("action", group[0].recommended_action or "Review evidence and adjust training.")
        impact = lit.get("impact", "")
        reference = lit.get("reference", "")

        finding_ids = [f"finding_{i}" for i, f in enumerate(findings) if f in group]

        rec = Recommendation(
            title=title,
            scope=scope,
            why=why,
            action=action,
            expected_impact=f"[{confidence_label}] {impact}" if confidence_label else impact,
            confidence=best_confidence,
            priority=priority,
            linked_finding_ids=finding_ids,
            example_command=reference,
        )
        recs.append(rec)

    mean_q = xai_summary.get("mean_quality_score")
    if mean_q is not None and mean_q < 0.5 and "low_xai_quality" not in groups:
        lit = _LITERATURE["low_xai_quality"]
        recs.append(
            Recommendation(
                title="Improve XAI quality globally",
                scope="global",
                why=f"Mean XAI quality score is {mean_q:.2f}; model may focus on background.",
                action=lit["action"],
                expected_impact=f"[Likely] {lit['impact']}",
                confidence="medium",
                priority=6,
                example_command=lit["reference"],
            )
        )

    if acc is not None and acc < 0.7 and "low_overall_accuracy" not in groups:
        lit = _LITERATURE["low_overall_accuracy"]
        recs.append(
            Recommendation(
                title="Improve overall accuracy",
                scope="global",
                why=f"Accuracy {acc:.2f} is below 70%.",
                action=lit["action"],
                expected_impact=f"[Likely] {lit['impact']}",
                confidence="medium",
                priority=7,
                example_command=lit["reference"],
            )
        )

    recs.sort(key=lambda r: (r.priority, r.title))
    return recs[:max_items]
