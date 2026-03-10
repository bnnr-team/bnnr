"""Root-cause oriented findings from metrics, confusion, XAI, and failure data.

Findings of the same type are grouped into a single entry listing all affected
classes, so the report is compact and noise-free.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bnnr.analysis.schema import FailurePattern, Finding


def build_findings(
    report: Any,
    class_diagnostics: list[Any],
    true_dist: dict[str, int],
    pred_dist: dict[str, int],
) -> tuple[list[Finding], list[FailurePattern]]:
    """Build grouped findings and failure patterns.

    Findings of the same type (e.g. all zero-recall classes) are merged into one
    entry with multiple class_ids for compact display.
    """
    raw_findings: list[Finding] = []
    patterns: list[FailurePattern] = []

    confusion = getattr(report, "confusion", {}) or {}
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    xai_diagnoses = getattr(report, "xai_diagnoses", {}) or {}
    metrics = getattr(report, "metrics", {}) or {}

    _detect_recall_issues(class_diagnostics, raw_findings, patterns)
    _detect_confused_pairs(matrix, labels_list, raw_findings, patterns)
    _detect_xai_issues(xai_diagnoses, raw_findings, patterns)
    _detect_global_issues(metrics, pred_dist, true_dist, raw_findings, patterns)

    grouped = _group_findings(raw_findings)

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    grouped.sort(key=lambda f: (sev_order.get(f.severity, 9), f.title))
    return grouped[:12], patterns


def _detect_recall_issues(
    class_diagnostics: list[Any],
    findings: list[Finding],
    patterns: list[FailurePattern],
) -> None:
    for d in class_diagnostics:
        if d.support == 0:
            continue
        if d.recall <= 0:
            findings.append(Finding(
                title="Zero recall",
                finding_type="zero_recall_class",
                description=f"Class {d.class_id}: no correct predictions (support={d.support}).",
                evidence=["recall=0", f"support={d.support}"],
                severity="critical",
                confidence="high",
                class_ids=[d.class_id],
            ))
            patterns.append(FailurePattern(
                pattern_type="zero_recall_class",
                description=f"Class {d.class_id}: zero recall",
                severity="critical",
                count=d.support,
                class_a=d.class_id,
            ))
        elif d.recall < 0.2:
            findings.append(Finding(
                title="Near-zero recall",
                finding_type="near_zero_recall",
                description=f"Class {d.class_id}: recall={d.recall:.0%} (support={d.support}).",
                evidence=[f"recall={d.recall:.2f}", f"support={d.support}"],
                severity="high",
                confidence="high",
                class_ids=[d.class_id],
            ))
            patterns.append(FailurePattern(
                pattern_type="near_zero_recall",
                description=f"Class {d.class_id}: recall={d.recall:.2f}",
                severity="high",
                count=d.support,
                class_a=d.class_id,
            ))


def _detect_confused_pairs(
    matrix: Any,
    labels_list: list,
    findings: list[Finding],
    patterns: list[FailurePattern],
) -> None:
    if not isinstance(matrix, list) or not labels_list:
        return
    import numpy as np

    mat = np.asarray(matrix)
    n = mat.shape[0]
    pairs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] > 0:
                pairs.append((i, j, int(mat[i, j])))
    pairs.sort(key=lambda x: -x[2])

    for rank, (i, j, count) in enumerate(pairs[:8]):
        true_id = str(labels_list[i]) if i < len(labels_list) else str(i)
        pred_id = str(labels_list[j]) if j < len(labels_list) else str(j)
        patterns.append(FailurePattern(
            pattern_type="confused_pair",
            description=f"True {true_id} \u2192 Pred {pred_id}",
            severity="high" if count >= 10 else "medium",
            count=count,
            class_a=true_id,
            class_b=pred_id,
        ))
        if rank < 4:
            findings.append(Finding(
                title="Frequent confusion pair",
                finding_type="confused_pair",
                description=f"{true_id}\u2192{pred_id}: {count} confusions (#{rank+1} pair).",
                evidence=[f"count={count}"],
                severity="high" if count >= 10 else "medium",
                confidence="high",
                class_ids=[true_id, pred_id],
                recommended_action="See Confusion Analysis section for XAI comparison.",
            ))


def _detect_xai_issues(
    xai_diagnoses: dict[str, Any],
    findings: list[Finding],
    patterns: list[FailurePattern],
) -> None:
    for cls_id, diag in xai_diagnoses.items():
        q = diag.get("quality_score")
        breakdown = diag.get("quality_breakdown") or {}
        edge_score = breakdown.get("edge")
        focus_score = breakdown.get("focus")
        coverage_score = breakdown.get("coverage")

        if q is not None and q < 0.5:
            findings.append(Finding(
                title="Low XAI quality",
                finding_type="low_xai_quality",
                description=f"Class {cls_id}: XAI quality={q:.2f}.",
                evidence=[f"quality={q:.2f}"],
                severity="medium",
                confidence="medium",
                class_ids=[str(cls_id)],
            ))
            patterns.append(FailurePattern(
                pattern_type="low_xai_quality",
                description=f"Class {cls_id}: XAI quality={q:.2f}",
                severity="medium",
                class_a=str(cls_id),
            ))

        if edge_score is not None and edge_score < 0.4:
            findings.append(Finding(
                title="Background focus suspected",
                finding_type="background_focus_suspected",
                description=f"Class {cls_id}: high border activation (edge={edge_score:.2f}).",
                evidence=[f"edge_score={edge_score:.2f}"],
                severity="high",
                confidence="medium",
                class_ids=[str(cls_id)],
            ))
            patterns.append(FailurePattern(
                pattern_type="background_focus_suspected",
                description=f"Border-focused saliency for class {cls_id}",
                severity="high",
                class_a=str(cls_id),
                metadata={"edge_score": edge_score},
            ))

        if (
            focus_score is not None
            and coverage_score is not None
            and focus_score < 0.3
            and coverage_score > 0.4
        ):
            findings.append(Finding(
                title="Diffuse attention pattern",
                finding_type="artifact_focus_suspected",
                description=(
                    f"Class {cls_id}: attention is broad (focus={focus_score:.2f}, "
                    f"coverage={coverage_score:.2f})."
                ),
                evidence=[f"focus={focus_score:.2f}", f"coverage={coverage_score:.2f}"],
                severity="medium",
                confidence="medium",
                class_ids=[str(cls_id)],
            ))


def _detect_global_issues(
    metrics: dict[str, Any],
    pred_dist: dict[str, int],
    true_dist: dict[str, int],
    findings: list[Finding],
    patterns: list[FailurePattern],
) -> None:
    acc = metrics.get("accuracy")
    if acc is not None and acc < 0.5:
        findings.append(Finding(
            title="Overall accuracy is critically low",
            finding_type="low_overall_accuracy",
            description=f"Accuracy {acc:.0%} indicates serious model or data issues.",
            evidence=[f"accuracy={acc:.2f}"],
            severity="critical",
            confidence="high",
        ))

    total_pred = sum(pred_dist.values())
    if total_pred > 0:
        pred_max = max(pred_dist.values()) if pred_dist else 0
        if pred_max > total_pred * 0.5:
            dominant = next((c for c, v in pred_dist.items() if v == pred_max), "")
            findings.append(Finding(
                title="Possible class collapse",
                finding_type="class_collapse_suspected",
                description=f"Class {dominant} receives {pred_max/total_pred:.0%} of all predictions.",
                evidence=[f"pred_max={pred_max}", f"total={total_pred}"],
                severity="high",
                confidence="medium",
                class_ids=[dominant],
            ))
            patterns.append(FailurePattern(
                pattern_type="class_collapse_suspected",
                description=f"Dominant pred class: {dominant}",
                severity="high",
                count=pred_max,
                class_a=dominant,
            ))

    total_true = sum(true_dist.values())
    for cls_id, pred_count in pred_dist.items():
        true_count = true_dist.get(cls_id, 0)
        if pred_count >= 20 and pred_count > true_count * 3:
            findings.append(Finding(
                title="Over-predicted class",
                finding_type="dominant_prediction_bias",
                description=f"Class {cls_id}: {pred_count} predictions vs {true_count} true.",
                evidence=[f"pred={pred_count}", f"true={true_count}"],
                severity="high",
                confidence="medium",
                class_ids=[cls_id],
            ))
            patterns.append(FailurePattern(
                pattern_type="dominant_prediction_bias",
                description=f"Class {cls_id} strongly over-predicted.",
                severity="high",
                count=pred_count,
                class_a=cls_id,
            ))

    for cls_id, true_count in true_dist.items():
        if true_count > 0 and true_count <= max(5, int(0.01 * total_true)):
            findings.append(Finding(
                title="Minority class at risk",
                finding_type="minority_class_suppression",
                description=f"Class {cls_id}: rare ({true_count} samples).",
                evidence=[f"support={true_count}"],
                severity="medium",
                confidence="medium",
                class_ids=[cls_id],
            ))


def _group_findings(findings: list[Finding]) -> list[Finding]:
    """Merge findings of the same type into grouped entries."""
    groups: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        groups[f.finding_type].append(f)

    grouped: list[Finding] = []
    for ftype, members in groups.items():
        if len(members) == 1:
            grouped.append(members[0])
            continue

        all_classes: list[str] = []
        all_evidence: list[str] = []
        best_sev = "low"
        best_conf = "low"
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        conf_order = {"high": 0, "medium": 1, "low": 2}

        for m in members:
            all_classes.extend(m.class_ids)
            all_evidence.extend(m.evidence[:2])
            if sev_order.get(m.severity, 9) < sev_order.get(best_sev, 9):
                best_sev = m.severity
            if conf_order.get(m.confidence, 9) < conf_order.get(best_conf, 9):
                best_conf = m.confidence

        unique_classes = list(dict.fromkeys(all_classes))
        class_tag = ", ".join(unique_classes[:8])
        if len(unique_classes) > 8:
            class_tag += f" (+{len(unique_classes) - 8})"

        descs = [m.description for m in members[:3]]
        if len(members) > 3:
            descs.append(f"...and {len(members) - 3} more.")
        combined_desc = " ".join(descs)

        title = members[0].title
        grouped.append(Finding(
            title=title,
            finding_type=ftype,
            description=combined_desc,
            evidence=all_evidence[:6],
            severity=best_sev,
            confidence=best_conf,
            class_ids=unique_classes,
            recommended_action=members[0].recommended_action,
        ))

    return grouped
