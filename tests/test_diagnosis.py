"""Tests for bnnr.analysis.diagnosis (FIX-1-2)."""

from __future__ import annotations

import pytest

from bnnr.analysis.diagnosis import (
    AttentionRegime,
    Diagnosis,
    DiagnosisThresholds,
    MissingThresholdsError,
    diagnose,
)
from bnnr.analysis.saliency_stats import SaliencyStats

CALIBRATED = DiagnosisThresholds(
    concentration_lo=0.30,
    concentration_hi=0.60,
    border_mass_hi=0.35,
    perturbation_shift_hi=0.50,
    robustness_gap_hi=0.15,
    min_confidence=0.75,
)


def _stats(
    *,
    concentration: float,
    border_mass: float,
    shift: float | None = None,
    gini: float = 0.5,
) -> SaliencyStats:
    return SaliencyStats(
        concentration=concentration,
        gini=gini,
        border_mass=border_mass,
        resolution=(14, 14),
        perturbation_shift=shift,
        perturbation_fill="gaussian_blur" if shift is not None else None,
        n_maps=64,
    )


def _shortcut_stats() -> SaliencyStats:
    """Diffuse, border-heavy, unstable."""
    return _stats(concentration=0.10, border_mass=0.60, shift=0.80)


def _focused_stats() -> SaliencyStats:
    """Concentrated, central, stable."""
    return _stats(concentration=0.85, border_mass=0.10, shift=0.15)


class TestThresholdsAreMandatory:
    def test_uncalibrated_thresholds_refuse_to_run(self) -> None:
        with pytest.raises(MissingThresholdsError, match="calibrated thresholds"):
            diagnose(
                _shortcut_stats(),
                overall_acc=0.9,
                hard_quantile_acc=0.4,
                thresholds=DiagnosisThresholds(),
            )

    def test_every_required_threshold_defaults_to_none(self) -> None:
        blank = DiagnosisThresholds()
        assert set(blank.missing()) == set(DiagnosisThresholds.REQUIRED)

    def test_error_names_the_missing_fields(self) -> None:
        partial = DiagnosisThresholds(
            concentration_lo=0.3, concentration_hi=0.6, border_mass_hi=0.35
        )
        with pytest.raises(MissingThresholdsError) as excinfo:
            partial.require()
        message = str(excinfo.value)
        assert "perturbation_shift_hi" in message
        assert "robustness_gap_hi" in message
        assert "concentration_lo" not in message

    def test_error_points_at_the_calibration_doc(self) -> None:
        with pytest.raises(MissingThresholdsError, match="docs/diagnosis.md"):
            DiagnosisThresholds().require()

    def test_min_confidence_is_not_required(self) -> None:
        """A caller may want the regime without a policy for acting on it."""
        thresholds = DiagnosisThresholds(
            concentration_lo=0.3,
            concentration_hi=0.6,
            border_mass_hi=0.35,
            perturbation_shift_hi=0.5,
            robustness_gap_hi=0.15,
        )
        thresholds.require()  # must not raise

    def test_inverted_concentration_band_rejected(self) -> None:
        thresholds = DiagnosisThresholds(
            concentration_lo=0.8,
            concentration_hi=0.2,
            border_mass_hi=0.35,
            perturbation_shift_hi=0.5,
            robustness_gap_hi=0.15,
        )
        with pytest.raises(ValueError, match="must not exceed"):
            thresholds.require()


class TestRegimeRule:
    def test_diffuse_border_unstable_with_a_gap_means_icd(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        assert result.regime is AttentionRegime.SHORTCUT_SUSPECTED
        assert result.recommended == ("icd",)
        assert result.confidence == pytest.approx(1.0)

    def test_concentrated_central_stable_with_a_gap_means_aicd(self) -> None:
        result = diagnose(
            _focused_stats(), overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        assert result.regime is AttentionRegime.OBJECT_FOCUSED
        assert result.recommended == ("aicd",)
        assert result.confidence == pytest.approx(1.0)

    def test_no_robustness_gap_means_no_directed_intervention(self) -> None:
        """Even a textbook shortcut signature: with nothing failing, nothing to fix."""
        result = diagnose(
            _shortcut_stats(), overall_acc=0.90, hard_quantile_acc=0.88, thresholds=CALIBRATED
        )
        assert result.regime is AttentionRegime.UNSTRUCTURED
        assert result.recommended == ("church_noise",)
        assert "no robustness gap" in result.reason

    def test_concentration_inside_the_band_means_no_usable_structure(self) -> None:
        middling = _stats(concentration=0.45, border_mass=0.60, shift=0.80)
        result = diagnose(
            middling, overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        assert result.regime is AttentionRegime.UNSTRUCTURED
        assert "no usable structure" in result.reason

    @pytest.mark.parametrize("concentration", [0.30, 0.60])
    def test_band_edges_count_as_inside_it(self, concentration: float) -> None:
        edge = _stats(concentration=concentration, border_mass=0.60, shift=0.80)
        result = diagnose(edge, overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED)
        assert result.regime is AttentionRegime.UNSTRUCTURED

    def test_mixed_evidence_picks_the_better_supported_regime(self) -> None:
        """Diffuse and border-heavy but stable: 3 of 4 shortcut clauses hold."""
        mixed = _stats(concentration=0.10, border_mass=0.60, shift=0.20)
        result = diagnose(
            mixed, overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        assert result.regime is AttentionRegime.SHORTCUT_SUSPECTED
        assert result.confidence == pytest.approx(0.75)


class TestConfidence:
    def test_confidence_is_a_count_out_of_four(self) -> None:
        mixed = _stats(concentration=0.10, border_mass=0.10, shift=0.20)
        result = diagnose(
            mixed, overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        assert result.confidence in {0.0, 0.25, 0.5, 0.75, 1.0}

    def test_full_agreement_is_one(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        assert result.confidence == pytest.approx(1.0)

    def test_unstructured_confidence_distinguishes_firm_from_borderline(self) -> None:
        """A clear no-gap case and a near-miss must not report the same number."""
        firm = diagnose(
            _stats(concentration=0.45, border_mass=0.10, shift=0.20),
            overall_acc=0.90,
            hard_quantile_acc=0.89,
            thresholds=CALIBRATED,
        )
        borderline = diagnose(
            _shortcut_stats(), overall_acc=0.90, hard_quantile_acc=0.88, thresholds=CALIBRATED
        )
        assert firm.confidence != borderline.confidence

    def test_criteria_are_exposed_with_their_margins(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        names = [c.name for c in result.criteria]
        assert names == ["diffuse", "border_heavy", "unstable", "robustness_gap"]
        assert all(c.margin >= 0 for c in result.criteria)

    def test_margin_is_distance_to_the_cut_point(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        diffuse = next(c for c in result.criteria if c.name == "diffuse")
        assert diffuse.margin == pytest.approx(abs(0.10 - 0.30))


class TestTieBreaking:
    def test_a_tie_goes_to_the_regime_that_is_less_nearly_false(self) -> None:
        """Two clauses each: the winner is the one whose weakest clause has the
        larger margin, so the decision is reproducible rather than dict-ordered."""
        tied = _stats(concentration=0.05, border_mass=0.10, shift=0.20)
        result = diagnose(
            tied, overall_acc=0.90, hard_quantile_acc=0.40, thresholds=CALIBRATED
        )
        # shortcut: diffuse yes, border no, unstable no, gap yes  -> 2/4
        # focused:  concentrated no, central yes, stable yes, gap yes -> 3/4
        assert result.regime is AttentionRegime.OBJECT_FOCUSED

    def test_the_same_input_always_gives_the_same_regime(self) -> None:
        stats = _stats(concentration=0.05, border_mass=0.35, shift=0.50)
        first = diagnose(stats, overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED)
        second = diagnose(stats, overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED)
        assert first.regime is second.regime
        assert first.confidence == second.confidence


class TestUnmeasuredPerturbationShift:
    def test_absent_shift_reads_as_stable(self) -> None:
        """The conservative direction: sharpen rather than replace."""
        no_shift = _stats(concentration=0.85, border_mass=0.10, shift=None)
        result = diagnose(
            no_shift, overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        stable = next(c for c in result.criteria if c.name == "stable")
        assert stable.satisfied
        assert result.regime is AttentionRegime.OBJECT_FOCUSED


class TestRecord:
    def test_to_dict_is_flat_enough_for_a_run_record(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        record = result.to_dict()
        assert record["regime"] == "shortcut_suspected"
        assert record["recommended"] == ["icd"]
        assert record["robustness_gap"] == pytest.approx(0.5)
        assert record["stats"]["resolution"] == (14, 14)
        assert len(record["criteria"]) == 4

    def test_gap_is_derived_from_the_two_accuracies(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.82, hard_quantile_acc=0.31, thresholds=CALIBRATED
        )
        assert result.robustness_gap == pytest.approx(0.51)

    def test_reason_names_the_clauses_that_held(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        assert "diffuse" in result.reason
        assert "border_heavy" in result.reason

    def test_regime_is_a_str_enum_so_it_serialises(self) -> None:
        result = diagnose(
            _shortcut_stats(), overall_acc=0.9, hard_quantile_acc=0.4, thresholds=CALIBRATED
        )
        assert isinstance(result, Diagnosis)
        assert result.regime == "shortcut_suspected"


class TestNoMaskDerivedInputs:
    def test_diagnose_takes_only_saliency_stats_and_accuracies(self) -> None:
        """The assumption-class guarantee, enforced at the signature.

        A rule that consumed masks would stop being deployable, so the only
        inputs are things BNNR computes from images and labels alone.
        """
        import inspect

        params = set(inspect.signature(diagnose).parameters)
        assert params == {"stats", "overall_acc", "hard_quantile_acc", "thresholds"}
