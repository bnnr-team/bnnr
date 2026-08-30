"""Tests for the deprecated XAI selection knobs (FIX-2-4)."""

from __future__ import annotations

import warnings

import pytest

from bnnr.config import apply_xai_preset, get_xai_preset, list_xai_presets
from bnnr.config_model import BNNRConfig
from bnnr.xai_analysis import compute_xai_quality_score


def _stats(n: int = 4, **overrides: float) -> list[dict[str, float]]:
    base = {
        "gini": 0.5,
        "coverage": 0.15,
        "spatial_coherence": 0.8,
        "edge_ratio": 0.1,
        "entropy": 5.0,
    }
    base.update(overrides)
    return [dict(base) for _ in range(n)]


class TestDeprecationWarnings:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("xai_selection_weight", 0.1), ("xai_pruning_threshold", 0.15)],
    )
    def test_setting_a_deprecated_knob_warns(self, field: str, value: float) -> None:
        with pytest.warns(DeprecationWarning, match=field):
            BNNRConfig(**{field: value})

    def test_the_warning_names_what_to_use_instead(self) -> None:
        with pytest.warns(DeprecationWarning, match="selector='diagnosis'"):
            BNNRConfig(xai_selection_weight=0.1)

    def test_the_pruning_warning_names_the_metric_based_knob(self) -> None:
        with pytest.warns(DeprecationWarning, match="candidate_pruning_relative_threshold"):
            BNNRConfig(xai_pruning_threshold=0.2)

    def test_not_setting_the_field_does_not_warn(self) -> None:
        """A user who never touched the knob is never nagged about it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            BNNRConfig()

    @pytest.mark.parametrize(
        ("field", "value"),
        [("xai_selection_weight", 0.0), ("xai_pruning_threshold", 0.0)],
    )
    def test_explicitly_disabling_does_not_warn(self, field: str, value: float) -> None:
        """Setting it to zero is turning the feature off, not relying on it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            BNNRConfig(**{field: value})

    def test_the_knob_still_works(self) -> None:
        """We stay in 0.x: deprecated is not removed."""
        with pytest.warns(DeprecationWarning):
            config = BNNRConfig(xai_selection_weight=0.3)
        assert config.xai_selection_weight == pytest.approx(0.3)


class TestPresetsShipNoSelectionWeight:
    @pytest.mark.parametrize("preset", ["xai_light", "xai_full", "xai_adaptive"])
    def test_no_preset_ships_an_uncalibrated_selection_weight(self, preset: str) -> None:
        assert get_xai_preset(preset)["xai_selection_weight"] == 0.0

    @pytest.mark.parametrize("preset", ["xai_light", "xai_full", "xai_adaptive"])
    def test_applying_a_preset_leaves_selection_on_the_metric(self, preset: str) -> None:
        config = apply_xai_preset(BNNRConfig(), preset)
        assert config.xai_selection_weight == 0.0

    def test_every_preset_is_covered_by_these_tests(self) -> None:
        """So a preset added later cannot quietly reintroduce a weight."""
        assert set(list_xai_presets()) == {"xai_light", "xai_full", "xai_adaptive"}

    def test_the_reporting_features_are_untouched(self) -> None:
        """Zeroing the weight must not turn off what the preset is for."""
        config = apply_xai_preset(BNNRConfig(), "xai_full")
        assert config.xai_enabled is True
        assert config.dual_xai_report is True
        assert config.adaptive_icd_threshold is True


class TestAccuracyIsNoLongerDoubleCounted:
    def test_accuracy_does_not_move_the_score(self) -> None:
        """It used to carry 25 %, and select_best_path then blended the score
        against the normalised selection metric, which is accuracy."""
        all_wrong, _ = compute_xai_quality_score(_stats(), [False] * 4)
        all_right, _ = compute_xai_quality_score(_stats(), [True] * 4)
        assert all_wrong == pytest.approx(all_right)

    def test_accuracy_is_still_reported_in_the_breakdown(self) -> None:
        """The dashboard displays it; nothing multiplies it."""
        _, breakdown = compute_xai_quality_score(_stats(), [True, True, False, False])
        assert breakdown["accuracy"] == pytest.approx(0.5)

    def test_the_score_still_spans_zero_to_one(self) -> None:
        worst, _ = compute_xai_quality_score(
            _stats(gini=0.0, coverage=0.9, spatial_coherence=0.0, edge_ratio=0.5),
            [True] * 4,
        )
        best, _ = compute_xai_quality_score(
            _stats(gini=1.0, coverage=0.15, spatial_coherence=1.0, edge_ratio=0.0),
            [True] * 4,
        )
        assert 0.0 <= worst < best <= 1.0

    def test_a_perfect_shape_scores_one(self) -> None:
        """The surviving weights are renormalised, so the top of the range is
        still reachable rather than capped at 0.75."""
        score, _ = compute_xai_quality_score(
            _stats(gini=1.0, coverage=0.15, spatial_coherence=1.0, edge_ratio=0.0),
            [True] * 4,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_saliency_shape_still_moves_the_score(self) -> None:
        """Removing accuracy must not make the score constant."""
        diffuse, _ = compute_xai_quality_score(_stats(gini=0.1), [True] * 4)
        sharp, _ = compute_xai_quality_score(_stats(gini=0.9), [True] * 4)
        assert sharp > diffuse

    def test_empty_stats_still_score_zero(self) -> None:
        score, breakdown = compute_xai_quality_score([], [])
        assert score == 0.0
        assert breakdown == {}

    def test_the_breakdown_keeps_every_component(self) -> None:
        _, breakdown = compute_xai_quality_score(_stats(), [True] * 4)
        assert set(breakdown) == {
            "accuracy", "focus", "coverage", "coherence", "edge", "consistency",
        }
