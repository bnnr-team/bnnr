"""Root-cause oriented findings from metrics, confusion, XAI, and failure data."""

from __future__ import annotations

from typing import Any

from bnnr.analysis.schema import FailurePattern, Finding


def build_findings(
    report: Any,
    class_diagnostics: list[Any],
    true_dist: dict[str, int],
    pred_dist: dict[str, int],
) -> tuple[list[Finding], list[FailurePattern]]:
    """Build structured findings and extended failure patterns from report data.

    report: AnalysisReport (or object with metrics, confusion, xai_diagnoses, failure_patterns).
    class_diagnostics: list of ClassDiagnostic from class_diagnostics.compute_class_diagnostics.
    Returns (findings, extended_failure_patterns).
    """
    findings: list[Finding] = []
    patterns: list[FailurePattern] = []

    confusion = getattr(report, "confusion", {}) or {}
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    xai_diagnoses = getattr(report, "xai_diagnoses", {}) or {}
    metrics = getattr(report, "metrics", {}) or {}
    worst_predictions = getattr(report, "worst_predictions", []) or []

    # --- Zero / near-zero recall classes ---
    for d in class_diagnostics:
        if d.support == 0:
            continue
        if d.recall <= 0:
            findings.append(
                Finding(
                    title=f"Class {d.class_id} has zero recall",
                    finding_type="zero_recall_class",
                    description=f"No sample of true class {d.class_id} was predicted correctly.",
                    evidence=[f"Support: {d.support}", "Correct: 0"],
                    interpretation="Model never predicts this class; likely under-represented or confused with others.",
                    severity="critical",
                    confidence="high",
                    class_ids=[d.class_id],
                    recommended_action="Add data or augmentation for this class; check confusion row for dominant predicted class.",
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="zero_recall_class",
                    description=f"Class {d.class_id}: zero recall",
                    severity="critical",
                    count=d.support,
                    class_a=d.class_id,
                    evidence=[f"support={d.support}"],
                    metadata={"class_id": d.class_id},
                )
            )
        elif d.recall < 0.2:
            findings.append(
                Finding(
                    title=f"Class {d.class_id} has very low recall ({d.recall:.0%})",
                    finding_type="near_zero_recall",
                    description=f"Only {d.recall:.0%} of class {d.class_id} samples are correct.",
                    evidence=[f"Recall: {d.recall:.2f}", f"Support: {d.support}"],
                    interpretation="Class is largely missed; check confusion for where predictions go.",
                    severity="high",
                    confidence="high",
                    class_ids=[d.class_id],
                    recommended_action="Consider more samples or targeted augmentation for this class.",
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="near_zero_recall",
                    description=f"Class {d.class_id}: recall={d.recall:.2f}",
                    severity="high",
                    count=d.support,
                    class_a=d.class_id,
                    metadata={"recall": d.recall, "support": d.support},
                )
            )

    # --- Confused pairs (from confusion matrix) ---
    if isinstance(matrix, list) and labels_list:
        import numpy as np
        mat = np.asarray(matrix)
        n = mat.shape[0]
        pairs: list[tuple[int, int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and mat[i, j] > 0:
                    pairs.append((i, j, int(mat[i, j])))
        pairs.sort(key=lambda x: -x[2])
        for i, j, count in pairs[:8]:
            true_id = str(labels_list[i]) if i < len(labels_list) else str(i)
            pred_id = str(labels_list[j]) if j < len(labels_list) else str(j)
            patterns.append(
                FailurePattern(
                    pattern_type="confused_pair",
                    description=f"True {true_id} → Pred {pred_id}",
                    severity="high" if count >= 10 else "medium",
                    count=count,
                    class_a=true_id,
                    class_b=pred_id,
                    evidence=[f"count={count}"],
                )
            )
            findings.append(
                Finding(
                    title=f"Model confuses class {true_id} with {pred_id}",
                    finding_type="confused_pair",
                    description=f"{count} samples of class {true_id} were predicted as {pred_id}.",
                    evidence=[f"Confusion count: {count}"],
                    interpretation="Possible semantic similarity or missing discriminative features.",
                    severity="high" if count >= 10 else "medium",
                    confidence="high",
                    class_ids=[true_id, pred_id],
                    recommended_action="Consider targeted augmentation or more data to separate these classes.",
                )
            )

    # --- Low XAI quality ---
    for cls_id, diag in xai_diagnoses.items():
        q = diag.get("quality_score")
        if q is not None and q < 0.5:
            findings.append(
                Finding(
                    title=f"Class {cls_id} has low XAI quality",
                    finding_type="low_xai_quality",
                    description=f"Saliency analysis suggests model may focus on background or artefacts for class {cls_id}.",
                    evidence=[f"XAI quality score: {q:.2f}"],
                    interpretation="Model may not be using class-specific features; consider ICD/AICD.",
                    severity="medium",
                    confidence="medium",
                    class_ids=[str(cls_id)],
                    recommended_action="Run BNNR train with XAI-guided augmentation (ICD or AICD) to encourage object focus.",
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="low_xai_quality",
                    description=f"Class {cls_id}: XAI quality={q:.2f}",
                    severity="medium",
                    class_a=str(cls_id),
                    metadata={"quality_score": q},
                )
            )

    # --- Overall accuracy / collapse ---
    acc = metrics.get("accuracy")
    if acc is not None and acc < 0.5:
        findings.append(
            Finding(
                title="Overall accuracy is low",
                finding_type="low_overall_accuracy",
                description=f"Accuracy {acc:.0%} indicates serious model or data issues.",
                evidence=[f"Accuracy: {acc:.2f}"],
                interpretation="Check class balance, data quality, and confusion matrix for dominant failure modes.",
                severity="critical",
                confidence="high",
                recommended_action="Review data quality, add augmentation, or collect more data.",
            )
        )

    total_pred = sum(pred_dist.values())
    total_true = sum(true_dist.values())
    if total_true > 0 and total_pred > 0:
        pred_max = max(pred_dist.values()) if pred_dist else 0
        if pred_max > total_pred * 0.5:
            dominant_class = next((c for c, v in pred_dist.items() if v == pred_max), "")
            findings.append(
                Finding(
                    title="Possible class collapse",
                    finding_type="class_collapse_suspected",
                    description=f"One class ({dominant_class}) receives >50% of all predictions.",
                    evidence=[f"Pred distribution max: {pred_max}", f"Total pred: {total_pred}"],
                    interpretation="Model may be collapsing to predicting one class; check confusion and data balance.",
                    severity="high",
                    confidence="medium",
                    class_ids=[dominant_class],
                    recommended_action="Check data balance and confusion matrix; consider class weights or augmentation.",
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="class_collapse_suspected",
                    description=f"Dominant pred class: {dominant_class}",
                    severity="high",
                    count=pred_max,
                    class_a=dominant_class,
                    metadata={"pred_max": pred_max, "total_pred": total_pred},
                )
            )

    # --- Dominant prediction bias (over-predicted classes) ---
    for cls_id, pred_count in pred_dist.items():
        true_count = true_dist.get(cls_id, 0)
        if pred_count >= 20 and pred_count > true_count * 3:
            findings.append(
                Finding(
                    title=f"Class {cls_id} is heavily over-predicted",
                    finding_type="dominant_prediction_bias",
                    description=(
                        f"Class {cls_id} appears in predictions far more often than in ground truth."
                    ),
                    evidence=[
                        f"Pred count: {pred_count}",
                        f"True count: {true_count}",
                    ],
                    interpretation=(
                        "Model may rely on generic background cues or biased features "
                        "that trigger this class too often."
                    ),
                    severity="high",
                    confidence="medium",
                    class_ids=[cls_id],
                    recommended_action=(
                        "Inspect XAI overlays and samples predicted as this class; "
                        "consider rebalancing data or adjusting decision thresholds."
                    ),
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="dominant_prediction_bias",
                    description=f"Class {cls_id} strongly over-predicted.",
                    severity="high",
                    count=pred_count,
                    class_a=cls_id,
                    metadata={"pred_count": pred_count, "true_count": true_count},
                )
            )

    # --- Minority class suppression (rare + low recall) ---
    for d in class_diagnostics:
        if d.support > 0 and d.support <= max(5, int(0.01 * total_true)) and d.recall < 0.5:
            findings.append(
                Finding(
                    title=f"Minority class {d.class_id} is suppressed",
                    finding_type="minority_class_suppression",
                    description=(
                        f"Class {d.class_id} is rare in the dataset and has low recall ({d.recall:.0%})."
                    ),
                    evidence=[f"Support: {d.support}", f"Recall: {d.recall:.2f}"],
                    interpretation=(
                        "Rare classes are not being learned reliably; the model may prioritise majority classes."
                    ),
                    severity="high",
                    confidence="medium",
                    class_ids=[d.class_id],
                    recommended_action=(
                        "Consider oversampling, class-balanced loss, or targeted augmentation "
                        "for this minority class."
                    ),
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="minority_class_suppression",
                    description=f"Minority class {d.class_id} underperforming.",
                    severity="high",
                    count=d.support,
                    class_a=d.class_id,
                    metadata={"recall": d.recall, "support": d.support},
                )
            )

    # --- High-confidence wrong vs low-confidence ambiguous (global) ---
    if worst_predictions:
        high_conf_wrong = [
            p for p in worst_predictions if p.get("confidence", 0.0) >= 0.8
        ]
        low_conf_wrong = [
            p
            for p in worst_predictions
            if 0.2 <= p.get("confidence", 0.0) < 0.8
        ]
        if high_conf_wrong:
            findings.append(
                Finding(
                    title="Many high-confidence wrong predictions",
                    finding_type="high_confidence_wrong",
                    description=(
                        f"{len(high_conf_wrong)} of the worst predictions have confidence ≥ 0.8."
                    ),
                    evidence=[
                        f"High-confidence wrong count among worst: {len(high_conf_wrong)}",
                        f"Total worst inspected: {len(worst_predictions)}",
                    ],
                    interpretation=(
                        "Model is overconfident on some failures; this is risky for production "
                        "and suggests miscalibration or spurious cues."
                    ),
                    severity="high",
                    confidence="medium",
                    recommended_action=(
                        "Inspect these cases with XAI; consider calibration (temperature scaling) "
                        "and revisiting training data / augmentations."
                    ),
                )
            )
        if low_conf_wrong:
            findings.append(
                Finding(
                    title="Ambiguous low-confidence errors",
                    finding_type="low_confidence_ambiguity",
                    description=(
                        f"{len(low_conf_wrong)} of the worst predictions have intermediate confidence."
                    ),
                    evidence=[
                        f"Low-confidence wrong count among worst: {len(low_conf_wrong)}",
                        f"Total worst inspected: {len(worst_predictions)}",
                    ],
                    interpretation=(
                        "These cases may be inherently ambiguous or poorly represented; "
                        "manual review can reveal labelling issues or missing subclasses."
                    ),
                    severity="medium",
                    confidence="medium",
                    recommended_action=(
                        "Perform targeted error analysis and potential relabelling for these samples."
                    ),
                )
            )

    # --- Background / artefact focus suspected from XAI breakdown ---
    for cls_id, diag in xai_diagnoses.items():
        breakdown = diag.get("quality_breakdown") or {}
        edge_score = breakdown.get("edge")
        coverage_score = breakdown.get("coverage")
        focus_score = breakdown.get("focus")
        if edge_score is not None and edge_score < 0.4:
            findings.append(
                Finding(
                    title=f"Background focus suspected for class {cls_id}",
                    finding_type="background_focus_suspected",
                    description=(
                        f"XAI suggests high border/edge activation for class {cls_id}."
                    ),
                    evidence=[f"edge_score={edge_score:.2f}"],
                    interpretation=(
                        "Model may rely on padding or background artefacts instead of the object."
                    ),
                    severity="high",
                    confidence="medium",
                    class_ids=[str(cls_id)],
                    recommended_action=(
                        "Review overlays; consider random crop/padding changes and ICD/AICD augmentations."
                    ),
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="background_focus_suspected",
                    description=f"Border-focused saliency for class {cls_id}",
                    severity="high",
                    class_a=str(cls_id),
                    metadata={"edge_score": edge_score},
                )
            )
        if (
            focus_score is not None
            and coverage_score is not None
            and focus_score < 0.3
            and coverage_score > 0.4
        ):
            findings.append(
                Finding(
                    title=f"Diffuse attention pattern for class {cls_id}",
                    finding_type="artifact_focus_suspected",
                    description=(
                        f"Attention for class {cls_id} is broad and unfocused (low focus, high coverage)."
                    ),
                    evidence=[
                        f"focus_score={focus_score:.2f}",
                        f"coverage_score={coverage_score:.2f}",
                    ],
                    interpretation=(
                        "Model may be picking up multiple spurious regions; features are not clearly "
                        "tied to the object."
                    ),
                    severity="medium",
                    confidence="medium",
                    class_ids=[str(cls_id)],
                    recommended_action=(
                        "Use stronger object-focused augmentations and ensure training data contains "
                        "clean, centered examples."
                    ),
                )
            )
            patterns.append(
                FailurePattern(
                    pattern_type="artifact_focus_suspected",
                    description=f"Diffuse focus for class {cls_id}",
                    severity="medium",
                    class_a=str(cls_id),
                    metadata={
                        "focus_score": focus_score,
                        "coverage_score": coverage_score,
                    },
                )
            )

    return findings, patterns
