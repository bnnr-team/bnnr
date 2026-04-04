"""Structured recommendations derived from report data plus literature context.

``evidence_from_run`` holds metrics/findings from this evaluation.
``expected_impact`` / ``literature_note`` summarize published practice — not
guaranteed effects for the current model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bnnr.analysis.schema import Finding, Recommendation

_LITERATURE: dict[str, dict[str, str]] = {
    "zero_recall_class": {
        "action": (
            "Apply focal loss (Lin et al., ICCV 2017) or class-balanced sampling. "
            "BNNR's ICD/AICD can find augmentations that improve per-class saliency."
        ),
        "impact_qualitative": (
            "Literature often reports recall gains on tail classes with focal loss "
            "or rebalancing; magnitude depends on the dataset."
        ),
        "reference": "Lin et al., Focal Loss for Dense Object Detection, ICCV 2017",
    },
    "class_collapse_suspected": {
        "action": (
            "Use deferred re-weighting or class-balanced sampling "
            "(Kang et al., ICLR 2020). Verify the dominant class isn't label-noisy."
        ),
        "impact_qualitative": (
            "Decoupled or reweighted training is a common fix when one class dominates "
            "predictions; verify on your validation split."
        ),
        "reference": "Kang et al., Decoupling Representation and Classifier, ICLR 2020",
    },
    "dominant_prediction_bias": {
        "action": (
            "Check XAI overlays for spurious correlations. Post-hoc temperature "
            "scaling (Guo et al., ICML 2017) can improve probability calibration."
        ),
        "impact_qualitative": (
            "Temperature scaling and related methods often lower ECE in published "
            "benchmarks; effect size varies."
        ),
        "reference": "Guo et al., On Calibration of Modern Neural Networks, ICML 2017",
    },
    "low_overall_accuracy": {
        "action": (
            "Review data quality — mislabeled samples degrade accuracy "
            "(Northcutt et al., JAIR 2021). Then try stronger augmentation or larger backbone."
        ),
        "impact_qualitative": (
            "Label cleaning and augmentation frequently help; reported gains are dataset-specific."
        ),
        "reference": "Northcutt et al., Confident Learning, JAIR 2021",
    },
    "near_zero_recall": {
        "action": (
            "Add targeted augmentation or oversample. Cost-sensitive learning "
            "with inverse-frequency weights (Buda et al., 2018) is a reliable first step."
        ),
        "impact_qualitative": (
            "Class weighting and oversampling are standard tools for rare classes; "
            "validate recall on a hold-out set."
        ),
        "reference": "Buda et al., A Systematic Study of the Class Imbalance Problem, 2018",
    },
    "minority_class_suppression": {
        "action": (
            "Apply class-balanced batch sampling and RandAugment (Cubuk et al., 2020) "
            "disproportionately on minority classes."
        ),
        "impact_qualitative": (
            "Balanced sampling plus strong augmentation is widely used for long-tail settings."
        ),
        "reference": "Cubuk et al., RandAugment, NeurIPS 2020",
    },
    "confused_pair": {
        "action": (
            "Inspect XAI overlays for both classes (see Confusion Analysis section). "
            "Pair-specific augmentation or metric learning (ArcFace; Deng et al., CVPR 2019) "
            "increases inter-class separation."
        ),
        "impact_qualitative": (
            "Targeted data collection or metric learning often reduces specific confusions; "
            "quantify with your confusion matrix after changes."
        ),
        "reference": "Deng et al., ArcFace, CVPR 2019",
    },
    "low_xai_quality": {
        "action": (
            "Run BNNR train with ICD/AICD augmentation search to encourage "
            "object-focused features."
        ),
        "impact_qualitative": (
            "Saliency quality scores are heuristics; combine with human review of overlays."
        ),
        "reference": "Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018",
    },
    "background_focus_suspected": {
        "action": (
            "Apply CutOut (DeVries & Taylor, 2017) or BNNR's ICD/AICD which "
            "penalizes background-focused saliency."
        ),
        "impact_qualitative": (
            "Spatial masking methods often reduce reliance on borders; confirm on your task."
        ),
        "reference": "DeVries & Taylor, Improved Regularization with Cutout, 2017",
    },
    "artifact_focus_suspected": {
        "action": (
            "Apply GridMask (Chen et al., 2020) or center-crop augmentation "
            "to force focus on a single discriminative region."
        ),
        "impact_qualitative": (
            "Structured masking can sharpen attention maps in some vision setups."
        ),
        "reference": "Chen et al., GridMask Data Augmentation, 2020",
    },
    "miscalibration_top1": {
        "action": (
            "Fit temperature scaling or vector scaling on a held-out split (Guo et al., ICML 2017). "
            "Recompute ECE after calibration."
        ),
        "impact_qualitative": (
            "Post-hoc calibration frequently reduces ECE in multiclass benchmarks; "
            "your measured ECE is the ground truth for this run."
        ),
        "reference": "Guo et al., On Calibration of Modern Neural Networks, ICML 2017",
    },
    "multilabel_fp_bias": {
        "action": (
            "Increase `multilabel_threshold` for the affected label(s), or tune per-label "
            "thresholds on validation precision–recall curves."
        ),
        "impact_qualitative": (
            "Threshold tuning trades precision vs recall per label; optimal values are data-dependent."
        ),
        "reference": "Standard multi-label decision thresholds (PR analysis)",
    },
    "multilabel_fn_bias": {
        "action": (
            "Decrease `multilabel_threshold` for the affected label(s), or add positives / "
            "class-weighted BCE for those heads."
        ),
        "impact_qualitative": (
            "Lowering thresholds typically raises recall at the cost of more false positives."
        ),
        "reference": "Standard multi-label decision thresholds (PR analysis)",
    },
    "low_multilabel_f1_macro": {
        "action": (
            "Review per-label support, consider asymmetric loss or focal BCE, and verify "
            "that label semantics match the head outputs."
        ),
        "impact_qualitative": (
            "Macro F1 aggregates across labels; inspect the worst labels in class diagnostics."
        ),
        "reference": "Multi-label classification best practices",
    },
}

_PRIORITY: dict[str, int] = {
    "zero_recall_class": 1,
    "class_collapse_suspected": 1,
    "low_multilabel_f1_macro": 1,
    "miscalibration_top1": 2,
    "dominant_prediction_bias": 2,
    "low_overall_accuracy": 2,
    "near_zero_recall": 3,
    "minority_class_suppression": 3,
    "multilabel_fn_bias": 3,
    "multilabel_fp_bias": 4,
    "confused_pair": 4,
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
    "low_xai_quality": "Improve saliency quality",
    "background_focus_suspected": "Fix background attention",
    "artifact_focus_suspected": "Fix diffuse attention patterns",
    "miscalibration_top1": "Improve probability calibration",
    "multilabel_fp_bias": "Reduce false positives per label",
    "multilabel_fn_bias": "Improve recall per label",
    "low_multilabel_f1_macro": "Raise macro F1 (multi-label)",
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
    """Build prioritized recommendations from findings."""
    metrics = getattr(report, "metrics", {}) or {}
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    class_diag = getattr(report, "class_diagnostics", []) or []
    acc = metrics.get("accuracy")

    diag_by_id: dict[str, dict[str, Any]] = {}
    for d in class_diag:
        cid = d.get("class_id", "") if isinstance(d, dict) else getattr(d, "class_id", "")
        diag_by_id[str(cid)] = d if isinstance(d, dict) else {}

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

        metric_details: list[str] = []
        for cid in unique_classes[:4]:
            d = diag_by_id.get(cid, {})
            rec_val = d.get("recall")
            prec_val = d.get("precision")
            f1_val = d.get("f1")
            if rec_val is not None:
                metric_details.append(f"label {cid} recall={float(rec_val):.0%}")
            elif prec_val is not None:
                metric_details.append(f"label {cid} precision={float(prec_val):.0%}")
            elif f1_val is not None:
                metric_details.append(f"label {cid} f1={float(f1_val):.2f}")

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

        evidence_from_finding: list[str] = []
        for f in group[:3]:
            evidence_from_finding.extend(f.evidence[:3])
        evidence_run = list(dict.fromkeys([*metric_details[:3], *evidence_from_finding[:4]]))

        why_parts = metric_details[:3] if metric_details else [f.description for f in group[:2]]
        why = ". ".join(why_parts) + "." if why_parts else group[0].description

        action = lit.get("action", group[0].recommended_action or "Review evidence and adjust training.")
        impact_lit = lit.get("impact_qualitative", "")
        reference = lit.get("reference", "")

        if impact_lit:
            expected = (
                f"[{confidence_label}] Literature context (not verified on this run): {impact_lit}"
                if confidence_label
                else f"Literature context (not verified on this run): {impact_lit}"
            )
        else:
            expected = f"[{confidence_label}] Review metrics and findings." if confidence_label else ""

        finding_ids = [f"finding_{i}" for i, f in enumerate(findings) if f in group]

        recs.append(
            Recommendation(
                title=title,
                scope=scope,
                why=why,
                action=action,
                expected_impact=expected,
                confidence=best_confidence,
                priority=priority,
                linked_finding_ids=finding_ids,
                example_command=reference,
                evidence_from_run=evidence_run,
                literature_note=reference,
            )
        )

    mean_q = xai_summary.get("mean_quality_score")
    if mean_q is not None and mean_q < 0.5 and "low_xai_quality" not in groups:
        lit = _LITERATURE["low_xai_quality"]
        iq = lit["impact_qualitative"]
        recs.append(
            Recommendation(
                title="Improve XAI quality globally",
                scope="global",
                why=f"Mean XAI quality score is {mean_q:.2f}.",
                action=lit["action"],
                expected_impact=f"[Likely] Literature context (not verified on this run): {iq}",
                confidence="medium",
                priority=6,
                example_command=lit["reference"],
                evidence_from_run=[f"mean_xai_quality={float(mean_q):.3f}"],
                literature_note=lit["reference"],
            )
        )

    if acc is not None and acc < 0.7 and "low_overall_accuracy" not in groups:
        lit = _LITERATURE["low_overall_accuracy"]
        iq = lit["impact_qualitative"]
        recs.append(
            Recommendation(
                title="Improve overall accuracy",
                scope="global",
                why=f"Accuracy {float(acc):.0%} is below 70%.",
                action=lit["action"],
                expected_impact=f"[Likely] Literature context (not verified on this run): {iq}",
                confidence="medium",
                priority=7,
                example_command=lit["reference"],
                evidence_from_run=[f"accuracy={float(acc):.3f}"],
                literature_note=lit["reference"],
            )
        )

    recs.sort(key=lambda r: (r.priority, r.title))
    return recs[:max_items]
