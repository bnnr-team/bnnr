"""Tests for the XAI Analysis module (bnnr.xai_analysis).

Covers:
- analyze_saliency_map: statistics from 2-D maps
- analyze_xai_batch: batched per-class insights
- generate_class_insight: text generation for different accuracy regimes
- generate_epoch_summary: epoch-level summaries
- get_analysis_metadata: lightweight aggregate stats
- compute_xai_quality_score: quality metric
- generate_class_diagnosis: enriched diagnosis with confusion + trends
- analyze_xai_batch_rich: enriched batch analysis
- generate_rich_epoch_summary: enriched epoch summary
- Edge cases: zero maps, single sample, mismatched sizes
"""

from __future__ import annotations

import numpy as np
import pytest

from bnnr.xai_analysis import (
    analyze_saliency_map,
    analyze_xai_batch,
    analyze_xai_batch_rich,
    compute_xai_quality_score,
    generate_class_diagnosis,
    generate_class_insight,
    generate_epoch_summary,
    generate_rich_epoch_summary,
    get_analysis_metadata,
)

# ---------------------------------------------------------------------------
#  analyze_saliency_map
# ---------------------------------------------------------------------------

class TestAnalyzeSaliencyMap:
    """Tests for single-map analysis."""

    def test_uniform_map(self) -> None:
        """A uniform (flat) map should have high entropy and centered mass."""
        m = np.ones((32, 32), dtype=np.float32) * 0.5
        stats = analyze_saliency_map(m)
        assert stats["entropy"] > 8.0
        assert abs(stats["center_x"] - 0.5) < 0.1
        assert abs(stats["center_y"] - 0.5) < 0.1
        assert stats["peak_value"] == pytest.approx(0.5)

    def test_zero_map(self) -> None:
        """An all-zero map should return safe defaults."""
        m = np.zeros((16, 16), dtype=np.float32)
        stats = analyze_saliency_map(m)
        assert stats["entropy"] == 0.0
        assert stats["center_x"] == 0.5
        assert stats["center_y"] == 0.5
        assert stats["coverage"] == 0.0

    def test_single_pixel_hotspot(self) -> None:
        """A single bright pixel should have low entropy and small coverage."""
        m = np.zeros((64, 64), dtype=np.float32)
        m[10, 20] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["entropy"] < 1.0  # Extremely focused
        assert stats["coverage"] < 0.01
        assert stats["peak_value"] == pytest.approx(1.0)
        # Center of mass should be near (20/63, 10/63)
        assert abs(stats["center_x"] - 20 / 63) < 0.02
        assert abs(stats["center_y"] - 10 / 63) < 0.02

    def test_top_left_hotspot(self) -> None:
        """Activity in the top-left quadrant should reflect in stats."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[:8, :8] = 1.0  # Top-left corner
        stats = analyze_saliency_map(m)
        assert stats["quadrant_tl"] > 0.9
        assert stats["center_x"] < 0.25
        assert stats["center_y"] < 0.25

    def test_bottom_right_hotspot(self) -> None:
        """Activity in the bottom-right quadrant."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[24:, 24:] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["quadrant_br"] > 0.9
        assert stats["center_x"] > 0.7
        assert stats["center_y"] > 0.7

    def test_coverage_half_active(self) -> None:
        """Coverage when half the image is above threshold."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[:16, :] = 1.0  # Top half
        stats = analyze_saliency_map(m)
        assert abs(stats["coverage"] - 0.5) < 0.05

    def test_returns_all_expected_keys(self) -> None:
        """Verify all expected keys are present."""
        m = np.random.rand(8, 8).astype(np.float32)
        stats = analyze_saliency_map(m)
        expected = {
            "entropy", "peak_value", "coverage",
            "center_x", "center_y",
            "quadrant_tl", "quadrant_tr", "quadrant_bl", "quadrant_br",
            "gini", "n_peaks", "spatial_coherence", "edge_ratio", "peak_to_mean",
        }
        assert set(stats.keys()) == expected

    # --- New metric tests ---

    def test_gini_focused(self) -> None:
        """A map with all mass on one pixel should have high Gini."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[16, 16] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["gini"] > 0.9

    def test_gini_uniform(self) -> None:
        """A uniform map should have near-zero Gini."""
        m = np.ones((32, 32), dtype=np.float32)
        stats = analyze_saliency_map(m)
        assert stats["gini"] < 0.1

    def test_n_peaks_single_blob(self) -> None:
        """A single bright blob should give n_peaks == 1."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[14:18, 14:18] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["n_peaks"] == 1.0

    def test_n_peaks_multiple_blobs(self) -> None:
        """Two separate bright blobs should give n_peaks >= 2."""
        m = np.zeros((64, 64), dtype=np.float32)
        m[5:10, 5:10] = 1.0
        m[50:55, 50:55] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["n_peaks"] >= 2.0

    def test_spatial_coherence_single_blob(self) -> None:
        """A single blob should have spatial_coherence near 1.0."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[10:20, 10:20] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["spatial_coherence"] > 0.8

    def test_edge_ratio_center(self) -> None:
        """Activation in the center should have low edge ratio."""
        m = np.zeros((64, 64), dtype=np.float32)
        m[20:44, 20:44] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["edge_ratio"] < 0.3

    def test_edge_ratio_border(self) -> None:
        """Activation on the border should have high edge ratio."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[0:3, :] = 1.0
        m[-3:, :] = 1.0
        m[:, 0:3] = 1.0
        m[:, -3:] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["edge_ratio"] > 0.5

    def test_peak_to_mean_focused(self) -> None:
        """A focused map should have high peak_to_mean."""
        m = np.zeros((32, 32), dtype=np.float32)
        m[16, 16] = 1.0
        stats = analyze_saliency_map(m)
        assert stats["peak_to_mean"] > 100.0

    def test_peak_to_mean_uniform(self) -> None:
        """A uniform map should have peak_to_mean == 1.0."""
        m = np.ones((32, 32), dtype=np.float32) * 0.5
        stats = analyze_saliency_map(m)
        assert abs(stats["peak_to_mean"] - 1.0) < 0.01


# ---------------------------------------------------------------------------
#  generate_class_insight
# ---------------------------------------------------------------------------

class TestGenerateClassInsight:
    """Tests for text insight generation."""

    def test_empty_stats(self) -> None:
        text = generate_class_insight("airplane", [], [])
        assert "No saliency data" in text

    def test_high_accuracy_focused(self) -> None:
        stats = [{"entropy": 6.0, "coverage": 0.05, "center_x": 0.5, "center_y": 0.5,
                  "peak_value": 1.0, "quadrant_tl": 0.25, "quadrant_tr": 0.25,
                  "quadrant_bl": 0.25, "quadrant_br": 0.25}]
        text = generate_class_insight("cat", stats, [True], xai_method="opticam")
        assert "sharply focused" in text
        assert "100%" in text
        assert "strong recognition" in text

    def test_low_accuracy_scattered(self) -> None:
        stats = [{"entropy": 14.0, "coverage": 0.6, "center_x": 0.5, "center_y": 0.5,
                  "peak_value": 0.8, "quadrant_tl": 0.25, "quadrant_tr": 0.25,
                  "quadrant_bl": 0.25, "quadrant_br": 0.25}]
        text = generate_class_insight("dog", stats, [False], xai_method="opticam")
        assert "scattered" in text
        assert "struggles" in text or "failed" in text

    def test_suggestion_for_broad_low_acc(self) -> None:
        """Should suggest ICD when coverage is broad and accuracy low."""
        stats = [{"entropy": 14.0, "coverage": 0.55, "center_x": 0.5, "center_y": 0.5,
                  "peak_value": 0.8, "quadrant_tl": 0.25, "quadrant_tr": 0.25,
                  "quadrant_bl": 0.25, "quadrant_br": 0.25}]
        text = generate_class_insight("ship", stats, [False], xai_method="opticam")
        assert "Suggestion:" in text
        assert "ICD" in text

    def test_moderate_accuracy(self) -> None:
        stats = [{"entropy": 10.0, "coverage": 0.15, "center_x": 0.5, "center_y": 0.5,
                  "peak_value": 0.9, "quadrant_tl": 0.4, "quadrant_tr": 0.1,
                  "quadrant_bl": 0.4, "quadrant_br": 0.1}]
        text = generate_class_insight("bird", stats, [True, True, False, True])
        assert "75%" in text
        assert "reasonably" in text

    def test_all_wrong_predictions(self) -> None:
        stats = [{"entropy": 7.0, "coverage": 0.03, "center_x": 0.1, "center_y": 0.1,
                  "peak_value": 1.0, "quadrant_tl": 0.95, "quadrant_tr": 0.02,
                  "quadrant_bl": 0.02, "quadrant_br": 0.01}]
        text = generate_class_insight("truck", stats, [False])
        assert "failed on all" in text or "0%" in text
        assert "irrelevant" in text

    def test_upper_left_location(self) -> None:
        stats = [{"entropy": 9.0, "coverage": 0.1, "center_x": 0.2, "center_y": 0.2,
                  "peak_value": 1.0, "quadrant_tl": 0.7, "quadrant_tr": 0.1,
                  "quadrant_bl": 0.1, "quadrant_br": 0.1}]
        text = generate_class_insight("deer", stats, [True])
        assert "upper-left" in text


# ---------------------------------------------------------------------------
#  compute_xai_quality_score
# ---------------------------------------------------------------------------

class TestComputeXAIQualityScore:
    """Tests for the scalar quality metric (returns (score, breakdown) tuple)."""

    def test_empty_stats(self) -> None:
        score, breakdown = compute_xai_quality_score([], [])
        assert score == 0.0
        assert breakdown == {}

    def test_perfect_score(self) -> None:
        """High accuracy + good Gini + moderate coverage + coherent → high score."""
        stats = [{
            "entropy": 4.0, "coverage": 0.10, "gini": 0.8,
            "spatial_coherence": 0.9, "edge_ratio": 0.05,
        }]
        score, breakdown = compute_xai_quality_score(stats, [True])
        assert score > 0.7
        assert "accuracy" in breakdown
        assert "focus" in breakdown
        assert "coverage" in breakdown
        assert "coherence" in breakdown
        assert "edge" in breakdown
        assert "consistency" in breakdown

    def test_poor_score(self) -> None:
        """Low accuracy + low Gini + broad coverage → low score."""
        stats = [{
            "entropy": 16.0, "coverage": 0.7, "gini": 0.1,
            "spatial_coherence": 0.1, "edge_ratio": 0.5,
        }]
        score, breakdown = compute_xai_quality_score(stats, [False])
        assert score < 0.3

    def test_score_in_range(self) -> None:
        """Score should always be in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(50):
            entropy = rng.uniform(0, 20)
            coverage = rng.uniform(0, 1)
            gini = rng.uniform(0, 1)
            sc = rng.uniform(0, 1)
            er = rng.uniform(0, 1)
            correct = bool(rng.rand() > 0.5)
            score, breakdown = compute_xai_quality_score(
                [{"entropy": entropy, "coverage": coverage, "gini": gini,
                  "spatial_coherence": sc, "edge_ratio": er}],
                [correct],
            )
            assert 0.0 <= score <= 1.0
            for v in breakdown.values():
                assert 0.0 <= v <= 1.0

    def test_mixed_accuracy(self) -> None:
        stats = [{
            "entropy": 10.0, "coverage": 0.15, "gini": 0.5,
            "spatial_coherence": 0.5, "edge_ratio": 0.2,
        }] * 4
        flags = [True, True, False, False]
        score, _ = compute_xai_quality_score(stats, flags)
        assert 0.2 < score < 0.8  # Moderate

    def test_breakdown_keys(self) -> None:
        """Breakdown dict should have exactly the expected 6 keys."""
        stats = [{"entropy": 8.0, "coverage": 0.15, "gini": 0.6,
                  "spatial_coherence": 0.7, "edge_ratio": 0.1}]
        _, breakdown = compute_xai_quality_score(stats, [True])
        expected_keys = {"accuracy", "focus", "coverage", "coherence", "edge", "consistency"}
        assert set(breakdown.keys()) == expected_keys

    def test_single_sample_consistency(self) -> None:
        """Single sample → consistency score should be 0.5 (neutral)."""
        stats = [{"entropy": 8.0, "coverage": 0.15, "gini": 0.5,
                  "spatial_coherence": 0.6, "edge_ratio": 0.1}]
        _, breakdown = compute_xai_quality_score(stats, [True])
        assert breakdown["consistency"] == 0.5

    def test_multi_sample_consistency(self) -> None:
        """Consistent samples (same entropy) → high consistency score."""
        stats = [
            {"entropy": 8.0, "coverage": 0.15, "gini": 0.5, "spatial_coherence": 0.6, "edge_ratio": 0.1},
            {"entropy": 8.1, "coverage": 0.14, "gini": 0.5, "spatial_coherence": 0.6, "edge_ratio": 0.1},
        ]
        _, breakdown = compute_xai_quality_score(stats, [True, True])
        assert breakdown["consistency"] > 0.9

    def test_edge_penalty(self) -> None:
        """High edge_ratio should result in a low edge sub-score."""
        stats = [{"entropy": 8.0, "coverage": 0.15, "gini": 0.5,
                  "spatial_coherence": 0.6, "edge_ratio": 0.7}]
        _, breakdown = compute_xai_quality_score(stats, [True])
        assert breakdown["edge"] < 0.1  # heavily penalized

    def test_coverage_bands(self) -> None:
        """Test that different coverage ranges produce different coverage sub-scores."""
        def _cov_score(cov: float) -> float:
            stats = [{"entropy": 8.0, "coverage": cov, "gini": 0.5,
                      "spatial_coherence": 0.6, "edge_ratio": 0.1}]
            _, b = compute_xai_quality_score(stats, [True])
            return b["coverage"]

        # <1% → 0.3; <5% → 0.6; 5-30% → 1.0; 30-50% → 0.5; >50% → 0.2
        assert _cov_score(0.005) == pytest.approx(0.3, abs=0.01)
        assert _cov_score(0.03) == pytest.approx(0.6, abs=0.01)
        assert _cov_score(0.15) == pytest.approx(1.0, abs=0.01)
        assert _cov_score(0.40) == pytest.approx(0.5, abs=0.01)
        assert _cov_score(0.60) == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
#  generate_class_diagnosis
# ---------------------------------------------------------------------------

class TestGenerateClassDiagnosis:
    """Tests for enriched diagnosis with confusion + trends + augmentation + baseline."""

    # Helper to create a full stats dict with all expected keys
    @staticmethod
    def _make_stats(
        entropy: float = 10.0, coverage: float = 0.15,
        gini: float = 0.5, spatial_coherence: float = 0.6,
        edge_ratio: float = 0.1, n_peaks: float = 1.0,
        peak_to_mean: float = 5.0,
        center_x: float = 0.5, center_y: float = 0.5,
        peak_value: float = 0.9,
    ) -> dict[str, float]:
        return {
            "entropy": entropy, "coverage": coverage,
            "center_x": center_x, "center_y": center_y,
            "peak_value": peak_value,
            "quadrant_tl": 0.25, "quadrant_tr": 0.25,
            "quadrant_bl": 0.25, "quadrant_br": 0.25,
            "gini": gini, "n_peaks": n_peaks,
            "spatial_coherence": spatial_coherence,
            "edge_ratio": edge_ratio, "peak_to_mean": peak_to_mean,
        }

    def test_empty_stats(self) -> None:
        result = generate_class_diagnosis("airplane", [], [])
        assert result["severity"] == "critical"
        assert result["quality_score"] == 0.0
        assert result["trend"] == "new"
        assert "No saliency data" in result["text"]
        assert "No data" in result["short_text"]
        assert result["quality_breakdown"] == {}
        assert result["augmentation_impact"] == ""
        assert result["baseline_delta"] == {}

    def test_returns_expected_keys(self) -> None:
        stats = [self._make_stats()]
        result = generate_class_diagnosis("cat", stats, [True])
        expected_keys = {
            "text", "severity", "quality_score", "quality_breakdown",
            "confused_with", "trend", "short_text",
            "augmentation_impact", "baseline_delta",
        }
        assert set(result.keys()) == expected_keys

    def test_ok_severity(self) -> None:
        """High accuracy + good quality → 'ok' severity."""
        stats = [self._make_stats(entropy=5.0, coverage=0.10, gini=0.8,
                                  spatial_coherence=0.9, edge_ratio=0.05)]
        result = generate_class_diagnosis("cat", stats, [True])
        assert result["severity"] == "ok"

    def test_critical_severity(self) -> None:
        """Low accuracy + poor focus → 'critical' severity."""
        stats = [self._make_stats(entropy=16.0, coverage=0.7, gini=0.1,
                                  spatial_coherence=0.1, edge_ratio=0.5)]
        result = generate_class_diagnosis("dog", stats, [False])
        assert result["severity"] == "critical"

    def test_confusion_pairs_included(self) -> None:
        """When confusion_pairs provided, they should appear in text and data."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis(
            "cat", stats, [True, False],
            confusion_pairs=[("dog", 5), ("tiger", 2)],
        )
        assert len(result["confused_with"]) == 2
        assert result["confused_with"][0]["class"] == "dog"
        assert result["confused_with"][0]["count"] == 5
        assert "dog" in result["text"]
        assert "dog" in result["short_text"]

    def test_trend_improving(self) -> None:
        """Entropy decrease → improving trend."""
        curr_stats = [self._make_stats(entropy=8.0, coverage=0.10)]
        prev_stats = [self._make_stats(entropy=12.0, coverage=0.15)]
        result = generate_class_diagnosis("cat", curr_stats, [True], prev_stats=prev_stats)
        assert result["trend"] == "improving"
        assert "sharpening" in result["text"]

    def test_trend_declining(self) -> None:
        """Entropy increase → declining trend."""
        curr_stats = [self._make_stats(entropy=14.0, coverage=0.40)]
        prev_stats = [self._make_stats(entropy=9.0, coverage=0.10)]
        result = generate_class_diagnosis("cat", curr_stats, [False], prev_stats=prev_stats)
        assert result["trend"] == "declining"
        assert "spreading" in result["text"]

    def test_trend_stable(self) -> None:
        """Minimal entropy change → stable trend."""
        stats = [self._make_stats(entropy=10.0, coverage=0.15)]
        prev_stats = [self._make_stats(entropy=10.2, coverage=0.16)]
        result = generate_class_diagnosis("cat", stats, [True], prev_stats=prev_stats)
        assert result["trend"] == "stable"

    def test_trend_new_without_prev(self) -> None:
        """No previous stats → 'new' trend."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis("cat", stats, [True])
        assert result["trend"] == "new"

    # --- Augmentation impact ---

    def test_augmentation_impact_with_changes(self) -> None:
        """When augmentation_name + prev_stats are given and there are deltas, augmentation_impact is generated."""
        curr = [self._make_stats(entropy=7.0, coverage=0.20, gini=0.7)]
        prev = [self._make_stats(entropy=10.0, coverage=0.10, gini=0.4)]
        result = generate_class_diagnosis(
            "cat", curr, [True],
            augmentation_name="aicd",
            prev_stats=prev,
        )
        assert "aicd" in result["augmentation_impact"]
        assert len(result["augmentation_impact"]) > 10

    def test_augmentation_impact_no_change(self) -> None:
        """When prev_stats is the same, augmentation_impact reports 'no significant changes'."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis(
            "cat", stats, [True],
            augmentation_name="icd",
            prev_stats=stats,
        )
        assert "no significant" in result["augmentation_impact"].lower()

    def test_augmentation_impact_empty_without_prev(self) -> None:
        """Without prev_stats, augmentation_impact should be empty."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis(
            "cat", stats, [True],
            augmentation_name="icd",
        )
        assert result["augmentation_impact"] == ""

    # --- Baseline delta ---

    def test_baseline_delta_computed(self) -> None:
        """When baseline_stats are provided, delta values should be computed."""
        curr = [self._make_stats(entropy=8.0, coverage=0.20, gini=0.7,
                                 spatial_coherence=0.8, edge_ratio=0.1)]
        bl = [self._make_stats(entropy=12.0, coverage=0.10, gini=0.4,
                               spatial_coherence=0.5, edge_ratio=0.2)]
        result = generate_class_diagnosis("cat", curr, [True], baseline_stats=bl)
        delta = result["baseline_delta"]
        assert "entropy" in delta
        assert "coverage_pp" in delta
        assert "gini" in delta
        assert "coherence" in delta
        assert "edge_ratio" in delta
        # Entropy decreased = negative delta
        assert delta["entropy"] < 0
        # Gini increased
        assert delta["gini"] > 0

    def test_baseline_delta_empty_without_baseline(self) -> None:
        """Without baseline_stats, delta should be empty."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis("cat", stats, [True])
        assert result["baseline_delta"] == {}

    # --- Quality breakdown ---

    def test_quality_breakdown_present(self) -> None:
        """quality_breakdown should be a dict with the expected keys."""
        stats = [self._make_stats()]
        result = generate_class_diagnosis("cat", stats, [True])
        bd = result["quality_breakdown"]
        assert isinstance(bd, dict)
        assert set(bd.keys()) == {"accuracy", "focus", "coverage", "coherence", "edge", "consistency"}

    # --- New metric-based suggestions ---

    def test_suggestion_scattered_low_coherence(self) -> None:
        """Low coherence + many peaks + <85% accuracy → suggest ICD."""
        stats = [self._make_stats(spatial_coherence=0.2, n_peaks=4.0,
                                  gini=0.3, coverage=0.45)]
        result = generate_class_diagnosis("cat", stats, [True, True, False, False])
        assert "Suggestions:" in result["text"]
        assert "ICD" in result["text"]

    def test_suggestion_high_edge_ratio(self) -> None:
        """High edge ratio + <90% accuracy → suggest padding augmentation."""
        stats = [self._make_stats(edge_ratio=0.45)]
        result = generate_class_diagnosis("cat", stats, [True, False])
        assert "border" in result["text"].lower() or "padding" in result["text"].lower()


# ---------------------------------------------------------------------------
#  analyze_xai_batch
# ---------------------------------------------------------------------------

class TestAnalyzeXAIBatch:
    """Tests for batched analysis."""

    def test_basic_batch(self) -> None:
        maps = np.random.rand(4, 16, 16).astype(np.float32)
        labels = [0, 1, 0, 1]
        preds = [0, 1, 1, 1]
        insights = analyze_xai_batch(maps, labels, preds)
        assert "0" in insights
        assert "1" in insights
        assert isinstance(insights["0"], str)
        assert len(insights["0"]) > 20  # Should be a non-trivial sentence

    def test_with_class_names(self) -> None:
        maps = np.random.rand(2, 8, 8).astype(np.float32)
        labels = [0, 1]
        insights = analyze_xai_batch(
            maps, labels, [0, 0],
            class_names=["airplane", "car"],
            xai_method="opticam",
        )
        assert "airplane" in insights["0"]
        assert "car" in insights["1"]

    def test_single_sample(self) -> None:
        maps = np.random.rand(1, 8, 8).astype(np.float32)
        insights = analyze_xai_batch(maps, [5], [5])
        assert "5" in insights

    def test_no_pred_labels(self) -> None:
        maps = np.random.rand(3, 8, 8).astype(np.float32)
        insights = analyze_xai_batch(maps, [0, 1, 2], None)
        assert len(insights) == 3

    def test_empty_batch(self) -> None:
        maps = np.zeros((0, 8, 8), dtype=np.float32)
        insights = analyze_xai_batch(maps, [], [])
        assert insights == {}

    def test_all_xai_methods_produce_compatible_maps(self) -> None:
        """Ensure the function handles maps from any BNNR XAI method.
        All methods produce [B, H, W] float32 maps, just different H/W.
        """
        for h, w in [(7, 7), (14, 14), (28, 28), (32, 32), (56, 56)]:
            maps = np.random.rand(4, h, w).astype(np.float32)
            insights = analyze_xai_batch(
                maps, [0, 1, 0, 1], [0, 1, 0, 1],
                xai_method="opticam",
            )
            assert len(insights) == 2
            for v in insights.values():
                assert isinstance(v, str)
                assert len(v) > 10


# ---------------------------------------------------------------------------
#  analyze_xai_batch_rich
# ---------------------------------------------------------------------------

class TestAnalyzeXAIBatchRich:
    """Tests for enriched batch analysis."""

    def test_basic_without_confusion(self) -> None:
        maps = np.random.rand(4, 16, 16).astype(np.float32)
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]
        insights, diagnoses, batch_stats = analyze_xai_batch_rich(
            maps, labels, preds,
        )
        assert "0" in insights
        assert "1" in insights
        assert "0" in diagnoses
        assert "1" in diagnoses
        assert "0" in batch_stats
        assert diagnoses["0"]["severity"] in ("ok", "warning", "critical")
        assert 0.0 <= diagnoses["0"]["quality_score"] <= 1.0

    def test_with_confusion_matrix(self) -> None:
        maps = np.random.rand(6, 8, 8).astype(np.float32)
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 1, 1, 2, 2, 0]
        confusion_matrix = [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
        insights, diagnoses, batch_stats = analyze_xai_batch_rich(
            maps, labels, preds,
            class_names=["cat", "dog", "bird"],
            confusion_matrix=confusion_matrix,
        )
        # Class 0 (cat) is confused with class 1 (dog) once
        confused_with_0 = diagnoses["0"]["confused_with"]
        assert any(c["class"] == "dog" for c in confused_with_0)

    def test_with_prev_batch_stats(self) -> None:
        """Passing previous stats should produce non-'new' trends."""
        maps = np.random.rand(4, 8, 8).astype(np.float32)
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        # Run twice — first produces 'new', second should have a real trend
        _, _, batch_stats_1 = analyze_xai_batch_rich(maps, labels, preds)
        _, diagnoses_2, _ = analyze_xai_batch_rich(
            maps, labels, preds,
            prev_batch_stats=batch_stats_1,
        )
        # When same maps are used, trend should be 'stable'
        for d in diagnoses_2.values():
            assert d["trend"] in ("improving", "stable", "declining")

    def test_empty_batch(self) -> None:
        maps = np.zeros((0, 8, 8), dtype=np.float32)
        insights, diagnoses, batch_stats = analyze_xai_batch_rich(maps, [], [])
        assert insights == {}
        assert diagnoses == {}
        assert batch_stats == {}

    def test_backward_compat_insights_match(self) -> None:
        """insights from rich should match basic analyze_xai_batch when no confusion/prev."""
        np.random.seed(42)
        maps = np.random.rand(4, 8, 8).astype(np.float32)
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 0]

        basic = analyze_xai_batch(maps, labels, preds, xai_method="opticam")
        rich_insights, _, _ = analyze_xai_batch_rich(
            maps, labels, preds, xai_method="opticam",
        )
        # Enriched insights text starts with the same base text
        # (may have extra confusion/trend info appended)
        for cls in basic:
            assert rich_insights[cls].startswith(basic[cls])

    def test_with_augmentation_name(self) -> None:
        """Passing augmentation_name + prev_batch_stats → augmentation_impact in diagnoses."""
        np.random.seed(200)
        maps1 = np.random.rand(4, 16, 16).astype(np.float32)
        maps2 = np.random.rand(4, 16, 16).astype(np.float32) * 0.5  # different maps
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]

        _, _, stats1 = analyze_xai_batch_rich(maps1, labels, preds)
        _, diagnoses2, _ = analyze_xai_batch_rich(
            maps2, labels, preds,
            prev_batch_stats=stats1,
            augmentation_name="icd",
        )
        # Each diagnosis should have augmentation_impact as a string
        for d in diagnoses2.values():
            assert isinstance(d["augmentation_impact"], str)
            assert "icd" in d["augmentation_impact"]

    def test_with_baseline_batch_stats(self) -> None:
        """Passing baseline_batch_stats → baseline_delta in diagnoses."""
        np.random.seed(300)
        maps_bl = np.random.rand(4, 16, 16).astype(np.float32)
        maps_curr = np.random.rand(4, 16, 16).astype(np.float32)
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]

        _, _, bl_stats = analyze_xai_batch_rich(maps_bl, labels, preds)
        _, diagnoses_curr, _ = analyze_xai_batch_rich(
            maps_curr, labels, preds,
            baseline_batch_stats=bl_stats,
        )
        for d in diagnoses_curr.values():
            assert isinstance(d["baseline_delta"], dict)
            # Should contain the expected delta keys
            if d["baseline_delta"]:
                assert "entropy" in d["baseline_delta"]
                assert "gini" in d["baseline_delta"]

    def test_diagnoses_have_quality_breakdown(self) -> None:
        """All diagnoses should contain quality_breakdown."""
        np.random.seed(400)
        maps = np.random.rand(4, 16, 16).astype(np.float32)
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        _, diagnoses, _ = analyze_xai_batch_rich(maps, labels, preds)
        for d in diagnoses.values():
            assert "quality_breakdown" in d
            assert isinstance(d["quality_breakdown"], dict)
            assert len(d["quality_breakdown"]) == 6  # 6 components


# ---------------------------------------------------------------------------
#  generate_epoch_summary
# ---------------------------------------------------------------------------

class TestGenerateEpochSummary:
    """Tests for epoch-level summaries."""

    def test_empty_insights(self) -> None:
        text = generate_epoch_summary({})
        assert "No XAI analysis available" in text

    def test_with_metrics(self) -> None:
        insights = {"0": "Some insight", "1": "Another insight"}
        metrics = {"accuracy": 0.85, "f1_macro": 0.82}
        text = generate_epoch_summary(
            insights,
            metrics,
            epoch=5,
            iteration=1,
            branch="smugs",
            xai_method="opticam",
        )
        assert "85.0%" in text
        assert "82.0%" in text
        assert "2 classes" in text
        assert "Opticam" in text

    def test_baseline_label(self) -> None:
        text = generate_epoch_summary(
            {"0": "x"},
            epoch=3,
            iteration=0,
            branch="baseline",
            xai_method="gradcam",
        )
        assert "baseline" in text


# ---------------------------------------------------------------------------
#  generate_rich_epoch_summary
# ---------------------------------------------------------------------------

class TestGenerateRichEpochSummary:
    """Tests for enriched epoch summaries."""

    def test_empty_diagnoses(self) -> None:
        text = generate_rich_epoch_summary({})
        assert "No XAI analysis available" in text

    def test_severity_counts(self) -> None:
        diagnoses = {
            "0": {"severity": "ok", "quality_score": 0.8, "trend": "stable"},
            "1": {"severity": "critical", "quality_score": 0.2, "trend": "declining"},
            "2": {"severity": "warning", "quality_score": 0.5, "trend": "improving"},
        }
        text = generate_rich_epoch_summary(
            diagnoses,
            {"accuracy": 0.75, "f1_macro": 0.70},
            epoch=5,
            iteration=1,
            branch="icd",
            xai_method="opticam",
        )
        assert "1 class(es) in critical state" in text
        assert "1 class(es) need attention" in text
        assert "quality score" in text.lower()

    def test_trend_counts(self) -> None:
        diagnoses = {
            "0": {"severity": "ok", "quality_score": 0.8, "trend": "improving"},
            "1": {"severity": "ok", "quality_score": 0.7, "trend": "improving"},
            "2": {"severity": "warning", "quality_score": 0.5, "trend": "declining"},
        }
        text = generate_rich_epoch_summary(diagnoses)
        assert "2 class(es) showing improved focus" in text
        assert "1 class(es) with declining focus" in text

    def test_with_metrics(self) -> None:
        diagnoses = {
            "0": {"severity": "ok", "quality_score": 0.9, "trend": "stable"},
        }
        text = generate_rich_epoch_summary(
            diagnoses,
            {"accuracy": 0.95, "f1_macro": 0.93},
            epoch=10,
            iteration=2,
            branch="smugs",
            xai_method="craft",
        )
        assert "95.0%" in text
        assert "Craft" in text


# ---------------------------------------------------------------------------
#  get_analysis_metadata
# ---------------------------------------------------------------------------

class TestGetAnalysisMetadata:
    """Tests for lightweight aggregate metadata."""

    def test_basic(self) -> None:
        maps = np.random.rand(4, 16, 16).astype(np.float32)
        meta = get_analysis_metadata(maps, [0, 1, 0, 1], [0, 0, 0, 1])
        assert meta["n_samples"] == 4
        assert meta["n_correct"] == 3  # indices 0, 2, 3
        assert 0 < meta["mean_entropy"]
        assert 0 < meta["mean_coverage"] <= 1.0

    def test_no_preds(self) -> None:
        maps = np.random.rand(2, 8, 8).astype(np.float32)
        meta = get_analysis_metadata(maps, [0, 1], None)
        assert meta["n_correct"] == 0
        assert meta["n_samples"] == 2

    def test_empty(self) -> None:
        maps = np.zeros((0, 8, 8), dtype=np.float32)
        meta = get_analysis_metadata(maps, [], [])
        assert meta["n_samples"] == 0
        assert meta["mean_entropy"] == 0.0


# ---------------------------------------------------------------------------
#  Integration: full pipeline from maps → insights
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end tests combining multiple functions."""

    def test_full_pipeline_opticam_like(self) -> None:
        """Simulate an OptiCAM-like pipeline with 7×7 saliency maps."""
        np.random.seed(42)
        batch_size = 8
        n_classes = 3
        maps = np.random.rand(batch_size, 7, 7).astype(np.float32)
        true_labels = [i % n_classes for i in range(batch_size)]
        pred_labels = [i % n_classes if i != 2 else (i + 1) % n_classes for i in range(batch_size)]
        class_names = ["airplane", "car", "dog"]

        insights = analyze_xai_batch(
            maps, true_labels, pred_labels,
            class_names=class_names,
            xai_method="opticam",
        )
        meta = get_analysis_metadata(maps, true_labels, pred_labels)
        summary = generate_epoch_summary(
            insights,
            {"accuracy": 0.875, "f1_macro": 0.85},
            epoch=10,
            iteration=2,
            branch="smugs",
            xai_method="opticam",
        )

        assert len(insights) == n_classes
        assert "airplane" in insights["0"]
        assert meta["n_samples"] == batch_size
        assert "3 classes" in summary
        assert "87.5%" in summary

    def test_full_pipeline_craft_like(self) -> None:
        """Simulate a CRAFT-like pipeline with 14×14 maps."""
        np.random.seed(123)
        maps = np.random.rand(6, 14, 14).astype(np.float32)
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 0, 1, 0, 2, 2]

        insights = analyze_xai_batch(
            maps, labels, preds,
            class_names=["cat", "dog", "bird"],
            xai_method="real_craft",
        )
        assert "0" in insights and "1" in insights and "2" in insights

    def test_full_pipeline_nmf_like(self) -> None:
        """Simulate NMF-based maps (typically same as activation map size)."""
        np.random.seed(456)
        maps = np.random.rand(4, 4, 4).astype(np.float32)
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]

        insights = analyze_xai_batch(
            maps, labels, preds,
            xai_method="nmf_concepts",
        )
        assert len(insights) == 2

    def test_rich_pipeline_with_confusion(self) -> None:
        """Full enriched pipeline: maps → rich analysis → rich summary."""
        np.random.seed(789)
        maps = np.random.rand(6, 16, 16).astype(np.float32)
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 1, 1, 2, 2, 0]
        class_names = ["cat", "dog", "bird"]
        confusion_matrix = [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]

        insights, diagnoses, batch_stats = analyze_xai_batch_rich(
            maps, labels, preds,
            class_names=class_names,
            xai_method="opticam",
            confusion_matrix=confusion_matrix,
        )
        summary = generate_rich_epoch_summary(
            diagnoses,
            {"accuracy": 0.5, "f1_macro": 0.45},
            epoch=5,
            iteration=1,
            branch="baseline",
            xai_method="opticam",
        )

        assert len(insights) == 3
        assert len(diagnoses) == 3
        assert len(batch_stats) == 3
        assert "3 classes" in summary
        # Should mention severity/quality
        assert "quality score" in summary.lower() or "critical" in summary.lower()

    def test_rich_pipeline_two_checkpoints_trend(self) -> None:
        """Two sequential rich analyses should produce meaningful trends."""
        np.random.seed(111)
        # First checkpoint: random maps
        maps1 = np.random.rand(4, 8, 8).astype(np.float32)
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]

        _, _, stats1 = analyze_xai_batch_rich(maps1, labels, preds)

        # Second checkpoint: same maps → trend should be 'stable'
        _, diagnoses2, _ = analyze_xai_batch_rich(
            maps1, labels, preds,
            prev_batch_stats=stats1,
        )
        for d in diagnoses2.values():
            assert d["trend"] == "stable"

    def test_rich_pipeline_full_features(self) -> None:
        """Full pipeline with confusion, prev_stats, augmentation, and baseline."""
        np.random.seed(999)
        n_classes = 3
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 0, 1, 1, 2, 2]
        class_names = ["cat", "dog", "bird"]
        confusion_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

        # Baseline
        maps_bl = np.random.rand(6, 16, 16).astype(np.float32)
        _, _, bl_stats = analyze_xai_batch_rich(
            maps_bl, labels, preds, class_names=class_names,
        )

        # Iteration 1
        maps_iter1 = np.random.rand(6, 16, 16).astype(np.float32)
        _, _, iter1_stats = analyze_xai_batch_rich(
            maps_iter1, labels, preds,
            class_names=class_names,
            confusion_matrix=confusion_matrix,
            baseline_batch_stats=bl_stats,
        )

        # Iteration 2 with augmentation
        maps_iter2 = np.random.rand(6, 16, 16).astype(np.float32)
        _, diagnoses_iter2, _ = analyze_xai_batch_rich(
            maps_iter2, labels, preds,
            class_names=class_names,
            confusion_matrix=confusion_matrix,
            prev_batch_stats=iter1_stats,
            baseline_batch_stats=bl_stats,
            augmentation_name="aicd",
        )

        assert len(diagnoses_iter2) == n_classes
        for cls_id, d in diagnoses_iter2.items():
            assert d["severity"] in ("ok", "warning", "critical")
            assert 0.0 <= d["quality_score"] <= 1.0
            assert len(d["quality_breakdown"]) == 6
            assert d["trend"] in ("improving", "stable", "declining")
            assert isinstance(d["augmentation_impact"], str)
            assert "aicd" in d["augmentation_impact"]
            assert isinstance(d["baseline_delta"], dict)
            if d["baseline_delta"]:
                assert "entropy" in d["baseline_delta"]
