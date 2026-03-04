"""XAI Analysis — lightweight human-readable explanations from saliency maps.

This module analyses pre-computed saliency maps (numpy arrays) using simple
statistical measures (entropy, center-of-mass, spatial quadrant distribution)
and produces short natural-language insights per class.  There are **no neural
networks or NLP models** involved — the text is generated via rule-based
templates, so the overhead is negligible (sub-millisecond on typical probe
sets of 10–40 images).

Works with every XAI method BNNR supports (OptiCAM / GradCAM, NMF, CRAFT,
RealCRAFT, RecursiveCRAFT) because all of them produce ``[B, H, W]``
saliency maps normalised to ``[0, 1]``.

Extended Analysis (v2)
----------------------
* :func:`compute_xai_quality_score` — scalar 0–1 quality metric per class.
* :func:`generate_class_diagnosis` — enriched per-class diagnosis that fuses
  saliency analysis with confusion-matrix data and cross-checkpoint trends.
* :func:`analyze_xai_batch_rich` — batch version returning structured
  diagnoses alongside classic text insights.
* :func:`generate_rich_epoch_summary` — epoch-level summary that highlights
  worst classes, severity counts, and trend indicators.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

__all__ = [
    "analyze_saliency_map",
    "analyze_xai_batch",
    "analyze_xai_batch_rich",
    "compute_xai_quality_score",
    "generate_class_diagnosis",
    "generate_class_insight",
    "generate_epoch_summary",
    "generate_rich_epoch_summary",
    "get_analysis_metadata",
]


# ---------------------------------------------------------------------------
#  Low-level map statistics
# ---------------------------------------------------------------------------

def analyze_saliency_map(map_2d: np.ndarray) -> dict[str, float]:
    """Compute focus statistics from a single ``[H, W]`` saliency map.

    Returns a dict with:
    * ``entropy`` — Shannon entropy of the map (treated as a probability
      distribution).  Lower = sharper focus.
    * ``peak_value`` — maximum activation value.
    * ``coverage`` — fraction of pixels above half-max (how much of the
      image is considered "important").
    * ``center_x``, ``center_y`` — centre-of-mass in ``[0, 1]`` coords.
    * ``quadrant_tl``, ``quadrant_tr``, ``quadrant_bl``, ``quadrant_br``
      — fraction of total activation in each spatial quadrant.
    * ``gini`` — Gini coefficient (0 = uniform, 1 = all mass in one pixel).
    * ``n_peaks`` — number of distinct activation blobs above 70 % of peak.
    * ``spatial_coherence`` — fraction of above-threshold mass in the
      largest connected component (1.0 = single blob).
    * ``edge_ratio`` — fraction of total activation in a 15 %-wide border
      strip.  High = model may attend to padding/border artefacts.
    * ``peak_to_mean`` — ``peak_value / mean_value``.  Higher = sharper.
    """
    h, w = map_2d.shape
    total = float(map_2d.sum())
    peak = float(map_2d.max())
    n_pixels = h * w

    if total <= 1e-8 or peak <= 1e-8:
        return {
            "entropy": 0.0,
            "peak_value": 0.0,
            "coverage": 0.0,
            "center_x": 0.5,
            "center_y": 0.5,
            "quadrant_tl": 0.25,
            "quadrant_tr": 0.25,
            "quadrant_bl": 0.25,
            "quadrant_br": 0.25,
            "gini": 0.0,
            "n_peaks": 0,
            "spatial_coherence": 0.0,
            "edge_ratio": 0.0,
            "peak_to_mean": 0.0,
        }

    prob = map_2d / total
    # Shannon entropy (base-2)
    entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))

    # Coverage: fraction of pixels above 50 % of peak
    coverage_mask = map_2d >= peak * 0.5
    coverage = float(np.mean(coverage_mask))

    # Centre of mass (normalised to [0, 1])
    y_grid, x_grid = np.indices((h, w))
    center_y = float(np.sum(y_grid * prob)) / max(h - 1, 1)
    center_x = float(np.sum(x_grid * prob)) / max(w - 1, 1)

    # Quadrant distribution
    mid_h, mid_w = h // 2, w // 2
    quadrant_tl = float(map_2d[:mid_h, :mid_w].sum()) / total
    quadrant_tr = float(map_2d[:mid_h, mid_w:].sum()) / total
    quadrant_bl = float(map_2d[mid_h:, :mid_w].sum()) / total
    quadrant_br = float(map_2d[mid_h:, mid_w:].sum()) / total

    # --- New metrics ---

    # Gini coefficient
    flat = np.sort(map_2d.ravel())
    index = np.arange(1, n_pixels + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(index * flat)) / (n_pixels * float(np.sum(flat))) - (n_pixels + 1) / n_pixels)
    gini = max(0.0, min(1.0, gini))

    # n_peaks: connected components above 70% of peak
    peak_mask = (map_2d >= peak * 0.7).astype(np.uint8)
    if peak_mask.any():
        n_labels, _ = cv2.connectedComponents(peak_mask, connectivity=8)
        n_peaks = max(n_labels - 1, 0)  # label 0 is background
    else:
        n_peaks = 0

    # Spatial coherence: fraction of coverage-mask pixels in largest CC
    cov_mask_u8 = coverage_mask.astype(np.uint8)
    cov_above = int(cov_mask_u8.sum())
    if cov_above > 0:
        n_labels_cov, labels_cov = cv2.connectedComponents(cov_mask_u8, connectivity=8)
        if n_labels_cov > 1:
            # Find largest non-background component
            component_sizes = np.bincount(np.asarray(labels_cov).ravel())
            component_sizes[0] = 0  # ignore background
            largest = int(component_sizes.max())
            spatial_coherence = float(largest) / float(cov_above)
        else:
            spatial_coherence = 0.0
    else:
        spatial_coherence = 0.0

    # Edge ratio: fraction of activation in 15%-border strip
    border_h = max(1, int(h * 0.15))
    border_w = max(1, int(w * 0.15))
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[border_h:h - border_h, border_w:w - border_w] = False
    edge_ratio = float(map_2d[border_mask].sum()) / total

    # Peak-to-mean ratio
    mean_val = total / n_pixels
    peak_to_mean = float(peak / mean_val) if mean_val > 1e-12 else 0.0

    return {
        "entropy": entropy,
        "peak_value": peak,
        "coverage": coverage,
        "center_x": center_x,
        "center_y": center_y,
        "quadrant_tl": quadrant_tl,
        "quadrant_tr": quadrant_tr,
        "quadrant_bl": quadrant_bl,
        "quadrant_br": quadrant_br,
        "gini": gini,
        "n_peaks": float(n_peaks),
        "spatial_coherence": spatial_coherence,
        "edge_ratio": edge_ratio,
        "peak_to_mean": peak_to_mean,
    }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _focus_descriptor(avg_entropy: float, avg_coverage: float) -> str:
    """Human-readable focus description."""
    if avg_entropy < 8.0:
        return "very sharply focused"
    if avg_entropy < 11.0:
        return "focused"
    if avg_entropy < 13.5:
        return "moderately spread"
    return "broadly scattered"


def _location_descriptor(cx: float, cy: float) -> str:
    """Describe spatial location from normalised centre-of-mass."""
    parts: list[str] = []
    if cy < 0.35:
        parts.append("upper")
    elif cy > 0.65:
        parts.append("lower")
    if cx < 0.35:
        parts.append("left")
    elif cx > 0.65:
        parts.append("right")
    if not parts:
        return "central"
    return "-".join(parts)


def _quadrant_summary(stats: list[dict[str, float]]) -> str:
    """Describe dominant quadrant from averaged stats."""
    keys = ["quadrant_tl", "quadrant_tr", "quadrant_bl", "quadrant_br"]
    labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
    avgs = [float(np.mean([s[k] for s in stats])) for k in keys]
    dominant_idx = int(np.argmax(avgs))
    dominant_val = avgs[dominant_idx]
    if dominant_val < 0.30:
        return "spread across the image (no dominant region)"
    return f"concentrated in the {labels[dominant_idx]} ({dominant_val * 100:.0f}%)"


# ---------------------------------------------------------------------------
#  Per-class insight generation (original — kept for backward compat)
# ---------------------------------------------------------------------------

def generate_class_insight(
    class_name: str,
    stats: list[dict[str, float]],
    correct_flags: list[bool],
    *,
    xai_method: str = "unknown",
) -> str:
    """Generate a concise, human-readable insight for one class.

    Parameters
    ----------
    class_name : str
        Display name (e.g. ``"airplane"`` or ``"class_3"``).
    stats : list[dict]
        Per-sample saliency statistics (from :func:`analyze_saliency_map`).
    correct_flags : list[bool]
        Whether each probe was predicted correctly.
    xai_method : str
        Name of the XAI method used (for context in the explanation).

    Returns
    -------
    str
        One-paragraph natural-language explanation.
    """
    if not stats:
        return f"No saliency data available for {class_name}."

    avg_entropy = float(np.mean([s["entropy"] for s in stats]))
    avg_coverage = float(np.mean([s["coverage"] for s in stats]))
    avg_cx = float(np.mean([s["center_x"] for s in stats]))
    avg_cy = float(np.mean([s["center_y"] for s in stats]))
    accuracy = float(np.mean(correct_flags)) * 100 if correct_flags else 0.0

    focus = _focus_descriptor(avg_entropy, avg_coverage)
    location = _location_descriptor(avg_cx, avg_cy)
    quadrant = _quadrant_summary(stats)

    lines: list[str] = []

    # Core focus description
    lines.append(
        f"The model's attention for \"{class_name}\" is {focus} "
        f"and primarily directed at the {location} area of the image."
    )

    # Quadrant detail
    if "no dominant" not in quadrant:
        lines.append(f"Most activation is {quadrant}.")

    # Coverage
    lines.append(
        f"About {avg_coverage * 100:.0f}% of the image is considered important "
        f"(coverage)."
    )

    # Prediction accuracy context
    if accuracy >= 95:
        lines.append(
            f"The model correctly identifies this class on {accuracy:.0f}% of "
            f"probe samples — strong recognition."
        )
    elif accuracy >= 70:
        lines.append(
            f"Probe accuracy is {accuracy:.0f}%. The model recognises this "
            f"class reasonably but may confuse it with visually similar classes."
        )
    elif accuracy > 0:
        lines.append(
            f"Probe accuracy is only {accuracy:.0f}%. The model struggles with "
            f"this class — the attention pattern may be unreliable or focusing "
            f"on background artefacts rather than the object itself."
        )
    else:
        lines.append(
            "The model failed on all probe samples for this class. "
            "It is likely attending to irrelevant features."
        )

    # Actionable hint
    if avg_coverage > 0.50 and accuracy < 70:
        lines.append(
            "Suggestion: the broad, unfocused attention pattern together with "
            "low accuracy may indicate the model relies on background context. "
            "Consider augmentations like ICD (Intelligent Coarse Dropout) to "
            "force the model to learn from the object itself."
        )
    elif avg_entropy > 13.0 and accuracy < 80:
        lines.append(
            "Suggestion: scattered attention with moderate accuracy could "
            "benefit from AICD or additional data augmentation to sharpen "
            "the model's focus."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
#  Quality score
# ---------------------------------------------------------------------------

def compute_xai_quality_score(
    stats: list[dict[str, float]],
    correct_flags: list[bool],
) -> tuple[float, dict[str, float]]:
    """Compute a 0–1 quality score from saliency statistics and accuracy.

    Higher score = better-focused model with higher accuracy.

    **6-component formula** (v2):

    =============================  ======  ================================
    Component                      Weight  Source
    =============================  ======  ================================
    accuracy                       25 %    fraction of correct predictions
    focus (Gini coefficient)       20 %    higher Gini → sharper focus
    coverage                       15 %    moderate (5–30 %) is ideal
    spatial coherence              15 %    single blob preferred
    edge ratio (inverted)          10 %    penalise border attention
    cross-sample consistency       15 %    low CV of entropy across samples
    =============================  ======  ================================

    Returns
    -------
    score : float
        Scalar quality score in ``[0, 1]``.
    breakdown : dict[str, float]
        Per-component sub-scores (each 0–1) for dashboard display.
    """
    if not stats:
        return 0.0, {}

    accuracy = float(np.mean(correct_flags)) if correct_flags else 0.0

    avg_gini = float(np.mean([s.get("gini", 0.0) for s in stats]))
    avg_coverage = float(np.mean([s.get("coverage", 0.0) for s in stats]))
    avg_coherence = float(np.mean([s.get("spatial_coherence", 0.0) for s in stats]))
    avg_edge = float(np.mean([s.get("edge_ratio", 0.0) for s in stats]))

    # Focus score: Gini in [0, 1]; higher = more concentrated
    focus_score = avg_gini

    # Coverage score — moderate is ideal
    if avg_coverage < 0.01:
        coverage_score = 0.3
    elif avg_coverage < 0.05:
        coverage_score = 0.6
    elif avg_coverage <= 0.30:
        coverage_score = 1.0
    elif avg_coverage <= 0.50:
        coverage_score = 0.5
    else:
        coverage_score = 0.2

    # Spatial coherence is already in [0, 1]
    coherence_score = avg_coherence

    # Edge ratio: lower is better  (0 → 1.0, 0.5 → 0.0)
    edge_score = max(0.0, 1.0 - avg_edge * 2.0)

    # Cross-sample consistency: low coefficient of variation of entropy
    entropies = [s.get("entropy", 0.0) for s in stats]
    if len(entropies) >= 2:
        mean_ent = float(np.mean(entropies))
        std_ent = float(np.std(entropies))
        cv = std_ent / max(mean_ent, 1e-8)
        consistency_score = max(0.0, 1.0 - cv)
    else:
        # Single sample — can't measure consistency, neutral 0.5
        consistency_score = 0.5

    breakdown = {
        "accuracy": round(accuracy, 4),
        "focus": round(focus_score, 4),
        "coverage": round(coverage_score, 4),
        "coherence": round(coherence_score, 4),
        "edge": round(edge_score, 4),
        "consistency": round(consistency_score, 4),
    }

    score = (
        accuracy * 0.25
        + focus_score * 0.20
        + coverage_score * 0.15
        + coherence_score * 0.15
        + edge_score * 0.10
        + consistency_score * 0.15
    )
    return round(min(1.0, max(0.0, score)), 4), breakdown


# ---------------------------------------------------------------------------
#  Enriched per-class diagnosis  (XAI + confusion + deltas)
# ---------------------------------------------------------------------------

def generate_class_diagnosis(
    class_name: str,
    stats: list[dict[str, float]],
    correct_flags: list[bool],
    *,
    xai_method: str = "unknown",
    confusion_pairs: list[tuple[str, int]] | None = None,
    prev_stats: list[dict[str, float]] | None = None,
    augmentation_name: str | None = None,
    baseline_stats: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Generate an enriched per-class diagnosis combining XAI, confusion, and trends.

    Parameters
    ----------
    class_name : str
        Display name of the class.
    stats : list[dict]
        Per-sample saliency statistics from :func:`analyze_saliency_map`.
    correct_flags : list[bool]
        Whether each probe was predicted correctly.
    xai_method : str
        XAI method name.
    confusion_pairs : list[(class_name, count)] | None
        Top classes this class is confused with, sorted by count desc.
        Extracted from the confusion matrix row for this class.
    prev_stats : list[dict] | None
        Saliency stats from the *previous* checkpoint for trend comparison.
    augmentation_name : str | None
        Name of the current augmentation (e.g. ``"aicd"``).  When provided
        together with *prev_stats*, generates an augmentation impact section.
    baseline_stats : list[dict] | None
        Saliency stats from the *baseline* checkpoint for delta-vs-baseline.

    Returns
    -------
    dict with keys:
        ``text``              — full human-readable diagnosis paragraph.
        ``severity``          — ``"ok"`` | ``"warning"`` | ``"critical"``.
        ``quality_score``     — float 0–1.
        ``quality_breakdown`` — per-component sub-scores dict.
        ``confused_with``     — list of ``{"class": str, "count": int}`` dicts.
        ``trend``             — ``"improving"`` | ``"stable"`` | ``"declining"`` | ``"new"``.
        ``short_text``        — one-line summary for compact display.
        ``augmentation_impact`` — text describing what the augmentation changed (or ``""``).
        ``baseline_delta``    — dict of metric deltas vs baseline (or ``{}``).
    """
    base_text = generate_class_insight(
        class_name, stats, correct_flags, xai_method=xai_method,
    )

    empty_breakdown: dict[str, float] = {}
    if not stats:
        return {
            "text": base_text,
            "severity": "critical",
            "quality_score": 0.0,
            "quality_breakdown": empty_breakdown,
            "confused_with": [],
            "trend": "new",
            "short_text": f"No data for {class_name}.",
            "augmentation_impact": "",
            "baseline_delta": {},
        }

    accuracy = float(np.mean(correct_flags)) * 100 if correct_flags else 0.0
    quality, quality_breakdown = compute_xai_quality_score(stats, correct_flags)

    # --- Severity ---
    if accuracy >= 90 and quality >= 0.6:
        severity = "ok"
    elif accuracy >= 60 or quality >= 0.4:
        severity = "warning"
    else:
        severity = "critical"

    # --- Confusion pairs ---
    confusion_lines: list[str] = []
    confused_with_data: list[dict[str, Any]] = []
    if confusion_pairs:
        for confused_class, count in confusion_pairs[:3]:
            confused_with_data.append({"class": confused_class, "count": count})
            confusion_lines.append(f"{confused_class} ({count}\u00d7)")

    # --- Trend from previous checkpoint ---
    trend = "new"
    trend_lines: list[str] = []
    if prev_stats and stats:
        prev_entropy = float(np.mean([s["entropy"] for s in prev_stats]))
        curr_entropy = float(np.mean([s["entropy"] for s in stats]))
        prev_coverage = float(np.mean([s["coverage"] for s in prev_stats]))
        curr_coverage = float(np.mean([s["coverage"] for s in stats]))

        entropy_delta = curr_entropy - prev_entropy
        coverage_delta = curr_coverage - prev_coverage

        if entropy_delta < -0.5:
            trend = "improving"
            trend_lines.append(
                f"Attention is sharpening (entropy {entropy_delta:+.1f})."
            )
        elif entropy_delta > 0.5:
            trend = "declining"
            trend_lines.append(
                f"Attention is spreading (entropy {entropy_delta:+.1f})."
            )
        else:
            trend = "stable"

        if abs(coverage_delta) > 0.05:
            direction = "expanded" if coverage_delta > 0 else "narrowed"
            trend_lines.append(
                f"Coverage {direction} by {abs(coverage_delta) * 100:.0f}pp."
            )

    # --- Augmentation impact ---
    augmentation_impact = ""
    if augmentation_name and prev_stats and stats:
        _prev_ent = float(np.mean([s["entropy"] for s in prev_stats]))
        _curr_ent = float(np.mean([s["entropy"] for s in stats]))
        _prev_cov = float(np.mean([s.get("coverage", 0) for s in prev_stats]))
        _curr_cov = float(np.mean([s.get("coverage", 0) for s in stats]))
        _prev_coh = float(np.mean([s.get("spatial_coherence", 0) for s in prev_stats]))
        _curr_coh = float(np.mean([s.get("spatial_coherence", 0) for s in stats]))
        _prev_gini = float(np.mean([s.get("gini", 0) for s in prev_stats]))
        _curr_gini = float(np.mean([s.get("gini", 0) for s in stats]))

        impact_parts: list[str] = [f"After applying {augmentation_name}:"]
        ent_delta = _curr_ent - _prev_ent
        if abs(ent_delta) > 0.3:
            verb = "sharpened" if ent_delta < 0 else "broadened"
            impact_parts.append(f"entropy {_prev_ent:.1f} → {_curr_ent:.1f} (attention {verb})")
        cov_delta = _curr_cov - _prev_cov
        if abs(cov_delta) > 0.03:
            verb = "expanded" if cov_delta > 0 else "narrowed"
            impact_parts.append(f"coverage {verb} {_prev_cov*100:.0f}% → {_curr_cov*100:.0f}%")
        coh_delta = _curr_coh - _prev_coh
        if abs(coh_delta) > 0.05:
            verb = "improved" if coh_delta > 0 else "declined"
            impact_parts.append(f"spatial coherence {verb} {_prev_coh:.2f} → {_curr_coh:.2f}")
        gini_delta = _curr_gini - _prev_gini
        if abs(gini_delta) > 0.03:
            verb = "improved" if gini_delta > 0 else "declined"
            impact_parts.append(f"focus (Gini) {verb} {_prev_gini:.2f} → {_curr_gini:.2f}")

        if len(impact_parts) > 1:
            augmentation_impact = "; ".join(impact_parts) + "."
        else:
            augmentation_impact = f"After applying {augmentation_name}: no significant saliency changes."

    # --- Baseline delta ---
    baseline_delta: dict[str, float] = {}
    baseline_lines: list[str] = []
    if baseline_stats and stats:
        bl_ent = float(np.mean([s["entropy"] for s in baseline_stats]))
        cu_ent = float(np.mean([s["entropy"] for s in stats]))
        bl_cov = float(np.mean([s.get("coverage", 0) for s in baseline_stats]))
        cu_cov = float(np.mean([s.get("coverage", 0) for s in stats]))
        bl_gini = float(np.mean([s.get("gini", 0) for s in baseline_stats]))
        cu_gini = float(np.mean([s.get("gini", 0) for s in stats]))
        bl_coh = float(np.mean([s.get("spatial_coherence", 0) for s in baseline_stats]))
        cu_coh = float(np.mean([s.get("spatial_coherence", 0) for s in stats]))
        bl_edge = float(np.mean([s.get("edge_ratio", 0) for s in baseline_stats]))
        cu_edge = float(np.mean([s.get("edge_ratio", 0) for s in stats]))

        baseline_delta = {
            "entropy": round(cu_ent - bl_ent, 2),
            "coverage_pp": round((cu_cov - bl_cov) * 100, 1),
            "gini": round(cu_gini - bl_gini, 3),
            "coherence": round(cu_coh - bl_coh, 3),
            "edge_ratio": round(cu_edge - bl_edge, 3),
        }

        delta_parts: list[str] = []
        if abs(baseline_delta["gini"]) > 0.02:
            direction = "improved" if baseline_delta["gini"] > 0 else "declined"
            delta_parts.append(f"focus {direction} by {abs(baseline_delta['gini']):.3f}")
        if abs(baseline_delta["coverage_pp"]) > 3:
            direction = "expanded" if baseline_delta["coverage_pp"] > 0 else "narrowed"
            delta_parts.append(f"coverage {direction} {abs(baseline_delta['coverage_pp']):.0f}pp")
        if abs(baseline_delta["coherence"]) > 0.05:
            direction = "+" if baseline_delta["coherence"] > 0 else ""
            delta_parts.append(f"coherence {direction}{baseline_delta['coherence']:.2f}")
        if delta_parts:
            baseline_lines.append("vs. baseline: " + ", ".join(delta_parts) + ".")

    # --- New-metric-based suggestions ---
    suggestion_parts: list[str] = []
    avg_coherence = float(np.mean([s.get("spatial_coherence", 0) for s in stats]))
    avg_n_peaks = float(np.mean([s.get("n_peaks", 1) for s in stats]))
    avg_edge = float(np.mean([s.get("edge_ratio", 0) for s in stats]))
    avg_gini = float(np.mean([s.get("gini", 0) for s in stats]))
    avg_coverage = float(np.mean([s.get("coverage", 0) for s in stats]))

    if avg_coherence < 0.4 and avg_n_peaks > 2 and accuracy < 85:
        suggestion_parts.append(
            "Model detects multiple scattered regions; ICD may help force "
            "single-object focus."
        )
    if avg_edge > 0.35 and accuracy < 90:
        suggestion_parts.append(
            "High border attention detected — model may rely on padding/border "
            "artefacts. Consider random-crop or padding augmentation."
        )
    if avg_gini < 0.3 and avg_coverage > 0.40 and accuracy < 80:
        suggestion_parts.append(
            "Attention is diffuse (low Gini, high coverage). Stronger "
            "augmentations like ICD or AICD could sharpen the model's focus."
        )
    if avg_gini > 0.85 and avg_coverage < 0.03:
        suggestion_parts.append(
            "Model is hyper-focused on a tiny region. This may indicate "
            "overfitting to a spurious feature. AICD may help diversify "
            "attention."
        )

    # --- Assemble enriched text ---
    parts: list[str] = [base_text]
    if confusion_lines:
        parts.append(
            f"Most confused with: {', '.join(confusion_lines)}."
        )
    if trend_lines:
        parts.append(" ".join(trend_lines))
    if augmentation_impact:
        parts.append(augmentation_impact)
    if baseline_lines:
        parts.append(" ".join(baseline_lines))
    if suggestion_parts:
        parts.append("Suggestions: " + " ".join(suggestion_parts))

    enriched_text = " ".join(parts)

    # --- Short one-liner ---
    avg_entropy = float(np.mean([s["entropy"] for s in stats]))
    focus_desc = _focus_descriptor(avg_entropy, avg_coverage)
    short = f"{class_name}: {focus_desc}, {accuracy:.0f}% accuracy"
    if confusion_lines:
        short += f" — confused with {confusion_lines[0]}"

    return {
        "text": enriched_text,
        "severity": severity,
        "quality_score": round(quality, 4),
        "quality_breakdown": quality_breakdown,
        "confused_with": confused_with_data,
        "trend": trend,
        "short_text": short,
        "augmentation_impact": augmentation_impact,
        "baseline_delta": baseline_delta,
    }


# ---------------------------------------------------------------------------
#  Batch analysis — original (backward compat)
# ---------------------------------------------------------------------------

def analyze_xai_batch(
    maps: np.ndarray,
    true_labels: list[int],
    pred_labels: list[int] | None = None,
    class_names: list[str] | None = None,
    xai_method: str = "unknown",
) -> dict[str, str]:
    """Generate per-class XAI insights from a batch of saliency maps.

    Parameters
    ----------
    maps : ndarray, shape ``[B, H, W]``
        Saliency maps (float, typically in ``[0, 1]``).
    true_labels : list[int]
        Ground-truth class IDs for each sample.
    pred_labels : list[int] | None
        Model predictions (if available).
    class_names : list[str] | None
        Mapping from class index to human name.
    xai_method : str
        Name of the XAI method used.

    Returns
    -------
    dict[str, str]
        Mapping ``class_id`` (as string) → insight text.
    """
    by_class_stats: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_class_correct: dict[int, list[bool]] = defaultdict(list)

    for i in range(len(maps)):
        label = int(true_labels[i])
        sal = maps[i]
        if sal.ndim != 2:
            continue
        stats = analyze_saliency_map(sal)
        by_class_stats[label].append(stats)
        is_correct = (label == int(pred_labels[i])) if pred_labels is not None else True
        by_class_correct[label].append(is_correct)

    insights: dict[str, str] = {}
    for cls in sorted(by_class_stats.keys()):
        name = class_names[cls] if (class_names and cls < len(class_names)) else f"class_{cls}"
        insights[str(cls)] = generate_class_insight(
            name,
            by_class_stats[cls],
            by_class_correct[cls],
            xai_method=xai_method,
        )

    return insights


# ---------------------------------------------------------------------------
#  Enriched batch analysis (v2) — returns structured diagnoses
# ---------------------------------------------------------------------------

def _extract_confusion_pairs(
    cls: int,
    confusion_matrix: list[list[int]],
    class_names: list[str] | None,
) -> list[tuple[str, int]] | None:
    """Extract sorted (class_name, count) pairs from a confusion matrix row."""
    if cls >= len(confusion_matrix):
        return None
    row = confusion_matrix[cls]
    pairs: list[tuple[str, int]] = []
    for other_cls in range(len(row)):
        if other_cls == cls:
            continue
        count = int(row[other_cls])
        if count > 0:
            other_name = (
                class_names[other_cls]
                if (class_names and other_cls < len(class_names))
                else f"class_{other_cls}"
            )
            pairs.append((other_name, count))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs if pairs else None


def analyze_xai_batch_rich(
    maps: np.ndarray,
    true_labels: list[int],
    pred_labels: list[int] | None = None,
    class_names: list[str] | None = None,
    xai_method: str = "unknown",
    confusion_matrix: list[list[int]] | None = None,
    prev_batch_stats: dict[str, list[dict[str, float]]] | None = None,
    augmentation_name: str | None = None,
    baseline_batch_stats: dict[str, list[dict[str, float]]] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
    """Enriched batch analysis returning structured diagnosis per class.

    Parameters
    ----------
    maps : ndarray ``[B, H, W]``
        Saliency maps.
    true_labels, pred_labels, class_names, xai_method
        Same as :func:`analyze_xai_batch`.
    confusion_matrix : list[list[int]] | None
        Full ``[N, N]`` confusion matrix (row = true, col = predicted).
    prev_batch_stats : dict[str, list[dict]] | None
        Per-class saliency stats from the *previous* checkpoint, keyed by
        class ID string.  Used for trend comparison.
    augmentation_name : str | None
        Name of the current augmentation.  Passed through to
        :func:`generate_class_diagnosis` for augmentation-impact text.
    baseline_batch_stats : dict[str, list[dict]] | None
        Per-class saliency stats from the *baseline* checkpoint.  When
        provided, each diagnosis includes a delta-vs-baseline section.

    Returns
    -------
    insights : dict[str, str]
        Classic per-class insight text (backward compatible).
    diagnoses : dict[str, dict]
        Structured per-class diagnosis (severity, quality_score, …).
    batch_stats : dict[str, list[dict]]
        Per-class saliency stats for this batch (pass as
        ``prev_batch_stats`` to the next call for trend deltas).
    """
    by_class_stats: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_class_correct: dict[int, list[bool]] = defaultdict(list)

    for i in range(len(maps)):
        label = int(true_labels[i])
        sal = maps[i]
        if sal.ndim != 2:
            continue
        s = analyze_saliency_map(sal)
        by_class_stats[label].append(s)
        is_correct = (label == int(pred_labels[i])) if pred_labels is not None else True
        by_class_correct[label].append(is_correct)

    insights: dict[str, str] = {}
    diagnoses: dict[str, dict[str, Any]] = {}
    batch_stats_out: dict[str, list[dict[str, float]]] = {}

    for cls in sorted(by_class_stats.keys()):
        name = (
            class_names[cls]
            if (class_names and cls < len(class_names))
            else f"class_{cls}"
        )

        # Confusion pairs from matrix row
        confusion_pairs = (
            _extract_confusion_pairs(cls, confusion_matrix, class_names)
            if confusion_matrix
            else None
        )

        prev_stats = (prev_batch_stats or {}).get(str(cls))
        bl_stats = (baseline_batch_stats or {}).get(str(cls))

        diagnosis = generate_class_diagnosis(
            name,
            by_class_stats[cls],
            by_class_correct[cls],
            xai_method=xai_method,
            confusion_pairs=confusion_pairs,
            prev_stats=prev_stats,
            augmentation_name=augmentation_name,
            baseline_stats=bl_stats,
        )

        insights[str(cls)] = diagnosis["text"]
        diagnoses[str(cls)] = diagnosis
        batch_stats_out[str(cls)] = by_class_stats[cls]

    return insights, diagnoses, batch_stats_out


# ---------------------------------------------------------------------------
#  Epoch-level summary (original — kept for backward compat)
# ---------------------------------------------------------------------------

def generate_epoch_summary(
    insights: dict[str, str],
    metrics: dict[str, float] | None = None,
    *,
    epoch: int = 0,
    iteration: int = 0,
    branch: str = "baseline",
    xai_method: str = "unknown",
) -> str:
    """Create a one-paragraph epoch-level summary from per-class insights.

    This is a lightweight roll-up intended for display at the top of the
    dashboard XAI section.
    """
    n_classes = len(insights)
    if n_classes == 0:
        return "No XAI analysis available for this checkpoint."

    acc = metrics.get("accuracy", 0.0) if metrics else 0.0
    f1 = metrics.get("f1_macro", 0.0) if metrics else 0.0

    parts: list[str] = []
    parts.append(
        f"XAI analysis for {branch} (iteration {iteration}, epoch {epoch}) "
        f"covering {n_classes} classes."
    )

    if acc > 0:
        parts.append(
            f"Overall accuracy: {acc * 100:.1f}%, F1-macro: {f1 * 100:.1f}%."
        )

    method_label = xai_method.replace("_", " ").title()
    parts.append(f"Saliency method: {method_label}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
#  Enriched epoch summary (v2)
# ---------------------------------------------------------------------------

def generate_rich_epoch_summary(
    diagnoses: dict[str, dict[str, Any]],
    metrics: dict[str, float] | None = None,
    *,
    epoch: int = 0,
    iteration: int = 0,
    branch: str = "baseline",
    xai_method: str = "unknown",
) -> str:
    """Enhanced epoch summary with worst-class highlights, trends, and scores.

    Uses the structured diagnosis produced by :func:`analyze_xai_batch_rich`
    to provide a more informative epoch-level overview.
    """
    n_classes = len(diagnoses)
    if n_classes == 0:
        return "No XAI analysis available for this checkpoint."

    acc = metrics.get("accuracy", 0.0) if metrics else 0.0
    f1 = metrics.get("f1_macro", 0.0) if metrics else 0.0

    parts: list[str] = []
    parts.append(
        f"XAI analysis for {branch} (iteration {iteration}, epoch {epoch}) "
        f"covering {n_classes} classes."
    )

    if acc > 0:
        parts.append(
            f"Overall accuracy: {acc * 100:.1f}%, F1-macro: {f1 * 100:.1f}%."
        )

    method_label = xai_method.replace("_", " ").title()
    parts.append(f"Saliency method: {method_label}.")

    # Severity counts
    critical = [cid for cid, d in diagnoses.items() if d.get("severity") == "critical"]
    warning = [cid for cid, d in diagnoses.items() if d.get("severity") == "warning"]

    if critical:
        parts.append(
            f"{len(critical)} class(es) in critical state "
            f"(low accuracy + poor focus)."
        )
    if warning:
        parts.append(f"{len(warning)} class(es) need attention.")

    # Average quality score
    scores = [d.get("quality_score", 0) for d in diagnoses.values()]
    if scores:
        avg_score = float(np.mean(scores))
        parts.append(f"Average XAI quality score: {avg_score:.2f}/1.00.")

    # Trend counts
    improving = sum(1 for d in diagnoses.values() if d.get("trend") == "improving")
    declining = sum(1 for d in diagnoses.values() if d.get("trend") == "declining")
    if improving > 0:
        parts.append(f"{improving} class(es) showing improved focus.")
    if declining > 0:
        parts.append(f"{declining} class(es) with declining focus quality.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
#  Lightweight aggregate metadata (unchanged)
# ---------------------------------------------------------------------------

def get_analysis_metadata(
    maps: np.ndarray,
    true_labels: list[int],
    pred_labels: list[int] | None = None,
) -> dict[str, Any]:
    """Return lightweight aggregate statistics for the whole batch.

    Intended for embedding in event payloads without the full text.
    """
    n_correct = 0
    n_total = len(true_labels)
    entropies: list[float] = []
    coverages: list[float] = []

    for i in range(len(maps)):
        sal = maps[i]
        if sal.ndim != 2:
            continue
        stats = analyze_saliency_map(sal)
        entropies.append(stats["entropy"])
        coverages.append(stats["coverage"])
        if pred_labels is not None and int(true_labels[i]) == int(pred_labels[i]):
            n_correct += 1

    return {
        "n_samples": n_total,
        "n_correct": n_correct,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
    }
