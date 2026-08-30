"""Tests for the indistinguishability test (FIX-2-3)."""

from __future__ import annotations

import numpy as np
import pytest

from bnnr.config_model import BNNRConfig
from bnnr.training.paired import (
    PairedInterval,
    correctness_vector,
    paired_bootstrap_ci,
)
from bnnr.training.selection import run_selector


def _config(**overrides) -> BNNRConfig:
    return BNNRConfig(**overrides)


class TestPairedBootstrap:
    def test_identical_arms_give_an_interval_containing_zero(self) -> None:
        same = np.array([True] * 60 + [False] * 40)
        interval = paired_bootstrap_ci(same, same, seed=0)
        assert interval is not None
        assert interval.difference == pytest.approx(0.0)
        assert interval.contains_zero

    def test_a_large_real_difference_excludes_zero(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        better = rng.random(n) < 0.90
        worse = rng.random(n) < 0.55
        interval = paired_bootstrap_ci(better, worse, seed=0)
        assert interval is not None
        assert not interval.contains_zero
        assert interval.low > 0

    def test_a_sub_noise_difference_contains_zero(self) -> None:
        """The Waterbirds shape: a spread smaller than the standard error."""
        rng = np.random.default_rng(1)
        n = 200
        base = rng.random(n) < 0.87
        nudged = base.copy()
        flip = rng.choice(np.flatnonzero(~base), size=1, replace=False)
        nudged[flip] = True  # one extra correct sample out of 200
        interval = paired_bootstrap_ci(nudged, base, seed=0)
        assert interval is not None
        assert interval.difference > 0  # it *is* larger
        assert interval.contains_zero  # but not distinguishably so

    def test_the_pairing_is_preserved(self) -> None:
        """Both arms are resampled with the same draw. Shuffling one arm
        independently would change the answer if they were not paired."""
        rng = np.random.default_rng(2)
        a = rng.random(300) < 0.8
        b = a.copy()
        paired = paired_bootstrap_ci(a, b, seed=0)
        assert paired is not None
        # Perfectly matched arms have a zero-width interval on the difference.
        assert paired.low == pytest.approx(0.0)
        assert paired.high == pytest.approx(0.0)

    def test_the_interval_is_reproducible_for_a_seed(self) -> None:
        rng = np.random.default_rng(3)
        a, b = rng.random(200) < 0.8, rng.random(200) < 0.75
        first = paired_bootstrap_ci(a, b, seed=7)
        second = paired_bootstrap_ci(a, b, seed=7)
        assert first == second

    def test_a_different_seed_moves_the_interval(self) -> None:
        rng = np.random.default_rng(4)
        a, b = rng.random(200) < 0.8, rng.random(200) < 0.75
        assert paired_bootstrap_ci(a, b, seed=1) != paired_bootstrap_ci(a, b, seed=2)

    def test_confidence_widens_the_interval(self) -> None:
        rng = np.random.default_rng(5)
        a, b = rng.random(300) < 0.8, rng.random(300) < 0.7
        narrow = paired_bootstrap_ci(a, b, confidence=0.80, seed=0)
        wide = paired_bootstrap_ci(a, b, confidence=0.99, seed=0)
        assert narrow is not None and wide is not None
        assert (wide.high - wide.low) > (narrow.high - narrow.low)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (np.array([True, False]), np.array([True])),  # mismatched lengths
            (np.array([True]), np.array([False])),  # fewer than two pairs
            (np.array([]), np.array([])),  # nothing at all
        ],
    )
    def test_unusable_input_returns_none_not_a_tie(self, a, b) -> None:
        """None means 'no test was possible', which is not the same as 'the
        arms tie' and must not be treated as one."""
        assert paired_bootstrap_ci(a, b) is None

    def test_integer_arrays_are_accepted(self) -> None:
        interval = paired_bootstrap_ci(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), seed=0)
        assert interval is not None
        assert interval.difference == pytest.approx(0.25)

    def test_to_dict_is_json_ready(self) -> None:
        import json

        interval = paired_bootstrap_ci(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), seed=0)
        assert interval is not None
        assert json.loads(json.dumps(interval.to_dict()))["n_pairs"] == 4


class TestCorrectnessVector:
    def test_builds_from_predictions_and_labels(self) -> None:
        got = correctness_vector(np.array([1, 2, 3]), np.array([1, 0, 3]))
        assert got is not None
        assert got.tolist() == [True, False, True]

    @pytest.mark.parametrize(
        ("preds", "labels"),
        [
            (None, np.array([1])),
            (np.array([1]), None),
            (np.array([1, 2]), np.array([1])),
            (np.array([]), np.array([])),
        ],
    )
    def test_missing_or_mismatched_input_is_none(self, preds, labels) -> None:
        assert correctness_vector(preds, labels) is None


class TestSelectionRefusesToSwitchOnATie:
    BASELINE = {"accuracy": 0.87}

    def _correct(self, n: int, accuracy: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random(n) < accuracy

    def test_a_synthetic_tie_reports_indistinguishable_and_keeps_baseline(self) -> None:
        base = self._correct(200, 0.87, seed=0)
        nudged = base.copy()
        nudged[np.flatnonzero(~base)[0]] = True

        result = run_selector(
            {"aug_a": {"accuracy": float(nudged.mean())}},
            {"accuracy": float(base.mean())},
            _config(),
            per_sample_correct={"aug_a": nudged},
            baseline_correct=base,
        )
        assert result.reason == "indistinguishable"
        assert result.selected == ()

    def test_a_real_improvement_is_still_selected(self) -> None:
        base = self._correct(400, 0.55, seed=1)
        better = self._correct(400, 0.90, seed=2)
        result = run_selector(
            {"aug_a": {"accuracy": float(better.mean())}},
            {"accuracy": float(base.mean())},
            _config(),
            per_sample_correct={"aug_a": better},
            baseline_correct=base,
        )
        assert result.reason == "improved"
        assert result.selected == ("aug_a",)

    def test_the_interval_is_attached_either_way(self) -> None:
        base = self._correct(300, 0.6, seed=3)
        better = self._correct(300, 0.85, seed=4)
        result = run_selector(
            {"aug_a": {"accuracy": float(better.mean())}},
            {"accuracy": float(base.mean())},
            _config(),
            per_sample_correct={"aug_a": better},
            baseline_correct=base,
        )
        assert isinstance(result.interval, PairedInterval)
        assert result.interval.n_pairs == 300

    def test_without_the_vectors_the_raw_comparison_stands(self) -> None:
        """A caller that never cached predictions behaves exactly as before."""
        result = run_selector(
            {"aug_a": {"accuracy": 0.8701}}, self.BASELINE, _config()
        )
        assert result.reason == "improved"
        assert result.selected == ("aug_a",)
        assert result.interval is None

    def test_a_candidate_below_baseline_is_rejected_before_the_test_runs(self) -> None:
        base = self._correct(200, 0.9, seed=5)
        worse = self._correct(200, 0.4, seed=6)
        result = run_selector(
            {"aug_a": {"accuracy": float(worse.mean())}},
            {"accuracy": float(base.mean())},
            _config(),
            per_sample_correct={"aug_a": worse},
            baseline_correct=base,
        )
        assert result.reason == "no_improvement"

    def test_every_selector_applies_the_test(self) -> None:
        """The gate is shared, so a contrast between selectors stays a contrast
        between their ranking rules."""
        base = self._correct(200, 0.87, seed=0)
        nudged = base.copy()
        nudged[np.flatnonzero(~base)[0]] = True
        for selector in ("metric_argmax", "random"):
            result = run_selector(
                {"aug_a": {"accuracy": float(nudged.mean())}},
                {"accuracy": float(base.mean())},
                _config(selector=selector),
                per_sample_correct={"aug_a": nudged},
                baseline_correct=base,
            )
            assert result.reason == "indistinguishable", selector


class TestConfig:
    def test_defaults(self) -> None:
        config = BNNRConfig()
        assert config.indistinguishable_resamples == 2000
        assert config.indistinguishable_confidence == pytest.approx(0.95)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
    def test_confidence_range_is_validated(self, bad: float) -> None:
        with pytest.raises(ValueError, match="indistinguishable_confidence"):
            BNNRConfig(indistinguishable_confidence=bad)

    def test_too_few_resamples_rejected(self) -> None:
        with pytest.raises(ValueError):
            BNNRConfig(indistinguishable_resamples=10)

    def test_the_configured_confidence_reaches_the_test(self) -> None:
        rng = np.random.default_rng(9)
        base = rng.random(300) < 0.80
        better = rng.random(300) < 0.86
        wide = run_selector(
            {"aug_a": {"accuracy": float(better.mean())}},
            {"accuracy": float(base.mean())},
            _config(indistinguishable_confidence=0.999),
            per_sample_correct={"aug_a": better},
            baseline_correct=base,
        )
        narrow = run_selector(
            {"aug_a": {"accuracy": float(better.mean())}},
            {"accuracy": float(base.mean())},
            _config(indistinguishable_confidence=0.50),
            per_sample_correct={"aug_a": better},
            baseline_correct=base,
        )
        assert wide.interval is not None and narrow.interval is not None
        assert (wide.interval.high - wide.interval.low) > (
            narrow.interval.high - narrow.interval.low
        )
