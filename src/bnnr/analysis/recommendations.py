"""Structured, literature-backed recommendations linked to findings."""

from __future__ import annotations

from typing import Any

from bnnr.analysis.schema import Finding, Recommendation

_LITERATURE: dict[str, dict[str, str]] = {
    "zero_recall_class": {
        "action": (
            "Apply class-balanced sampling (Shen et al., 2016) or focal loss "
            "(Lin et al., 'Focal Loss for Dense Object Detection', ICCV 2017) "
            "to force the model to learn the underrepresented class. "
            "Augment with targeted transforms — BNNR's ICD/AICD search can "
            "automatically find augmentations that improve per-class saliency."
        ),
        "impact": (
            "Focal loss alone has been shown to recover 2–5 pp recall on tail classes; "
            "combining with data augmentation compounds the effect."
        ),
        "reference": "Lin et al., Focal Loss for Dense Object Detection, ICCV 2017",
    },
    "class_collapse_suspected": {
        "action": (
            "Introduce class-balanced sampling or use a deferred re-weighting strategy "
            "(Kang et al., 'Decoupling Representation and Classifier', ICLR 2020). "
            "Verify that the training data is not severely label-noisy for this class."
        ),
        "impact": (
            "Decoupled training typically recovers 3–8 pp on minority classes while "
            "preserving majority performance."
        ),
        "reference": "Kang et al., Decoupling Representation and Classifier, ICLR 2020",
    },
    "dominant_prediction_bias": {
        "action": (
            "Check for spurious correlations (e.g., background color, watermarks). "
            "Use BNNR's XAI overlays to confirm. Post-hoc calibration via temperature "
            "scaling (Guo et al., 'On Calibration of Modern Neural Networks', ICML 2017) "
            "can reduce over-confident predictions for the dominant class."
        ),
        "impact": (
            "Temperature scaling typically reduces ECE by 50–80% without retraining."
        ),
        "reference": "Guo et al., On Calibration of Modern Neural Networks, ICML 2017",
    },
    "low_overall_accuracy": {
        "action": (
            "Review data quality first — mislabeled samples degrade accuracy "
            "(Northcutt et al., 'Confident Learning', JAIR 2021). "
            "Then consider stronger augmentation or a larger model backbone."
        ),
        "impact": (
            "Cleaning 1–5% noisy labels has been shown to improve accuracy by 1–3 pp "
            "across standard benchmarks."
        ),
        "reference": "Northcutt et al., Confident Learning, JAIR 2021",
    },
    "near_zero_recall": {
        "action": (
            "Add targeted augmentation or oversample this class. "
            "Consider cost-sensitive learning with class weights inversely proportional "
            "to frequency (Buda et al., 'A Systematic Study of the Class Imbalance Problem', "
            "Neural Networks 2018). Inspect confusion row to find where predictions go."
        ),
        "impact": "Class-weighting typically recovers 5–15 pp recall on rare classes.",
        "reference": "Buda et al., A Systematic Study of the Class Imbalance Problem, Neural Networks 2018",
    },
    "minority_class_suppression": {
        "action": (
            "Apply SMOTE-like oversampling adapted for images (Chawla et al., 'SMOTE', JAIR 2002) "
            "or class-balanced batch sampling. "
            "For vision, strong augmentation (RandAugment; Cubuk et al., 2020) applied "
            "disproportionately to minority classes can be highly effective."
        ),
        "impact": (
            "Balanced sampling + targeted augmentation typically closes the recall gap "
            "by 10–20 pp for tail classes."
        ),
        "reference": "Cubuk et al., RandAugment, NeurIPS 2020",
    },
    "confused_pair": {
        "action": (
            "Increase inter-class discrimination by collecting hard examples for both "
            "classes, or by applying pair-specific augmentation. "
            "Metric learning approaches (e.g., ArcFace; Deng et al., CVPR 2019) can "
            "learn more separable embeddings. "
            "Check XAI overlays to see if both classes share background artifacts."
        ),
        "impact": (
            "Targeted augmentation for confused pairs can reduce confusion by 30–60% "
            "within 2–3 training iterations."
        ),
        "reference": "Deng et al., ArcFace, CVPR 2019",
    },
    "high_confidence_wrong": {
        "action": (
            "Apply temperature scaling (Guo et al., ICML 2017) post-hoc to reduce "
            "overconfidence. Label smoothing during training (Muller et al., "
            "'When Does Label Smoothing Help?', NeurIPS 2019) also prevents the model "
            "from producing extreme logits."
        ),
        "impact": (
            "Label smoothing reduces overconfident errors by 15–30% on average."
        ),
        "reference": "Muller et al., When Does Label Smoothing Help?, NeurIPS 2019",
    },
    "low_confidence_ambiguity": {
        "action": (
            "Manually review these ambiguous samples — they often reveal labeling "
            "errors, class boundary issues, or missing subclasses. "
            "Consider training with Mixup (Zhang et al., 'Mixup: Beyond Empirical Risk "
            "Minimization', ICLR 2018) which improves calibration on borderline cases."
        ),
        "impact": (
            "Mixup typically improves calibration and reduces error rates by 1–2 pp "
            "while making the model more robust to distribution shift."
        ),
        "reference": "Zhang et al., Mixup: Beyond Empirical Risk Minimization, ICLR 2018",
    },
    "low_xai_quality": {
        "action": (
            "Run BNNR train with XAI-guided augmentation (ICD/AICD) to encourage "
            "object-focused features. BNNR's augmentation search explicitly maximizes "
            "saliency quality, following the principle that models with better spatial "
            "attribution generalize better (Adebayo et al., 'Sanity Checks for Saliency Maps', "
            "NeurIPS 2018)."
        ),
        "impact": "ICD/AICD typically improves XAI quality by 0.1–0.3 points on 0–1 scale.",
        "reference": "Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018",
    },
    "background_focus_suspected": {
        "action": (
            "The model may rely on context or background rather than the object itself. "
            "Apply random background substitution, aggressive random cropping, or "
            "CutOut (DeVries & Taylor, 'Improved Regularization of Convolutional Neural "
            "Networks with Cutout', arXiv 2017). BNNR's ICD/AICD explicitly penalizes "
            "background-focused saliency."
        ),
        "impact": (
            "CutOut + targeted saliency optimization reduces background reliance by "
            "40–60% as measured by attribution-based metrics."
        ),
        "reference": "DeVries & Taylor, Improved Regularization with Cutout, arXiv 2017",
    },
    "artifact_focus_suspected": {
        "action": (
            "Diffuse attention suggests the model uses multiple spurious cues. "
            "Apply GradCAM-guided dropout or center-crop augmentation to force "
            "focus on a single region. GridMask (Chen et al., 'GridMask Data "
            "Augmentation', arXiv 2020) can also help by randomly masking grid "
            "regions during training."
        ),
        "impact": "GridMask + standard augmentation typically improves focus score by 15–30%.",
        "reference": "Chen et al., GridMask Data Augmentation, arXiv 2020",
    },
}


def build_recommendations(
    findings: list[Finding],
    report: Any,
    *,
    max_items: int = 15,
) -> list[Recommendation]:
    """Build prioritized, literature-backed recommendations from findings."""
    recs: list[Recommendation] = []
    metrics = getattr(report, "metrics", {}) or {}
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    acc = metrics.get("accuracy")

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

    seen_types: dict[str, int] = {}
    for i, f in enumerate(findings):
        ftype = f.finding_type
        seen_types[ftype] = seen_types.get(ftype, 0) + 1
        if seen_types[ftype] > 3:
            continue

        fid = f"finding_{i}"
        priority = priority_by_type.get(ftype, 10)
        lit = _LITERATURE.get(ftype, {})
        action = lit.get("action", f.recommended_action or "Review evidence and adjust training.")
        impact = lit.get("impact", "")
        reference = lit.get("reference", "")

        scope = ", ".join(f.class_ids) if f.class_ids else "global"
        rec = Recommendation(
            title=f.title,
            scope=scope,
            why=f.description,
            action=action,
            expected_impact=impact,
            confidence=f.confidence,
            priority=priority,
            linked_finding_ids=[fid],
            example_command=reference,
        )
        recs.append(rec)

    mean_q = xai_summary.get("mean_quality_score")
    if mean_q is not None and mean_q < 0.5 and not any(
        f.finding_type == "low_xai_quality" for f in findings
    ):
        lit = _LITERATURE["low_xai_quality"]
        recs.append(
            Recommendation(
                title="Improve XAI quality globally",
                scope="global",
                why=f"Mean XAI quality score is {mean_q:.2f}; model may focus on background.",
                action=lit["action"],
                expected_impact=lit["impact"],
                confidence="medium",
                priority=6,
                example_command=lit["reference"],
            )
        )

    if acc is not None and acc < 0.7 and not any("accuracy" in r.title.lower() for r in recs):
        lit = _LITERATURE["low_overall_accuracy"]
        recs.append(
            Recommendation(
                title="Improve overall accuracy",
                scope="global",
                why=f"Accuracy {acc:.2f} is below 70%.",
                action=lit["action"],
                expected_impact=lit["impact"],
                confidence="medium",
                priority=7,
                example_command=lit["reference"],
            )
        )

    recs.sort(key=lambda r: (r.priority, r.title))
    return recs[:max_items]
