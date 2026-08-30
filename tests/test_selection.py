"""Tests for bnnr.training.selection (FIX-2-1)."""

from __future__ import annotations

import random

import pytest

from bnnr.config_model import BNNRConfig
from bnnr.training.branching import select_best_path
from bnnr.training.selection import (
    SELECTORS,
    CandidateReport,
    SelectionResult,
    get_selector,
    run_selector,
)


def _config(**overrides) -> BNNRConfig:
    return BNNRConfig(**overrides)


# ---------------------------------------------------------------------------
# The reference implementation, copied verbatim from branching.py as it stood
# before this change. The differential test below is the actual proof that
# metric_argmax is behaviour-preserving; hand-written cases only cover what
# someone thought to write down.
# ---------------------------------------------------------------------------


def _legacy_select_best_path(results, baseline_metrics, config, xai_scores=None):
    metric = config.selection_metric
    mode = config.selection_mode
    w = config.xai_selection_weight

    baseline_value = baseline_metrics.get(metric)

    if w <= 0 or not xai_scores:
        best_name = None
        best_value = None
        for aug_name, aug_metrics in results.items():
            val = aug_metrics.get(metric)
            if val is None:
                continue
            if best_value is None or (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
                best_name = aug_name
                best_value = val
        if best_name is None or baseline_value is None or best_value is None:
            return None
        improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
        return best_name if improved else None

    metric_vals = {name: m.get(metric) for name, m in results.items() if m.get(metric) is not None}
    if not metric_vals:
        return None

    all_vals = list(metric_vals.values())
    min_m = min(v for v in all_vals if v is not None)
    max_m = max(v for v in all_vals if v is not None)
    m_range = max_m - min_m if max_m != min_m else 1.0

    best_name = None
    best_composite = None
    for aug_name, val in metric_vals.items():
        if val is None:
            continue
        if mode == "max":
            norm_m = (float(val) - float(min_m)) / float(m_range)
        else:
            norm_m = (float(max_m) - float(val)) / float(m_range)
        xai_q = xai_scores.get(aug_name, 0.0)
        composite = (1.0 - w) * norm_m + w * xai_q
        if best_composite is None or composite > best_composite:
            best_composite = composite
            best_name = aug_name

    if best_name is None or baseline_value is None:
        return None

    best_value = results[best_name].get(metric)
    if best_value is None:
        return None
    improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
    return best_name if improved else None


class TestMetricArgmaxIsBehaviourPreserving:
    """The default path must return what it returned before the refactor."""

    @pytest.mark.parametrize("trial", range(300))
    def test_matches_the_legacy_implementation(self, trial: int) -> None:
        rng = random.Random(trial)
        mode = rng.choice(["max", "min"])
        weight = rng.choice([0.0, 0.0, 0.1, 0.5, 0.9, 1.0])
        config = _config(selection_mode=mode, xai_selection_weight=weight)

        n = rng.randint(1, 5)
        names = [f"aug_{i}" for i in range(n)]
        results = {}
        for name in names:
            if rng.random() < 0.15:
                results[name] = {}  # candidate that never reported the metric
            else:
                # Deliberately include near-ties and exact ties.
                results[name] = {"accuracy": round(rng.uniform(0.80, 0.86), rng.choice([2, 4]))}

        baseline = {"accuracy": round(rng.uniform(0.78, 0.86), 3)}
        if rng.random() < 0.1:
            baseline = {}

        xai_scores = None
        if rng.random() < 0.6:
            xai_scores = {name: round(rng.random(), 3) for name in names if rng.random() < 0.8}

        expected = _legacy_select_best_path(results, baseline, config, xai_scores)
        assert select_best_path(results, baseline, config, xai_scores) == expected

    def test_exact_tie_keeps_the_first_candidate(self) -> None:
        """Legacy used a strict >, so ties resolved to insertion order."""
        results = {"aug_a": {"accuracy": 0.9}, "aug_b": {"accuracy": 0.9}}
        baseline = {"accuracy": 0.5}
        assert select_best_path(results, baseline, _config()) == "aug_a"

    def test_non_empty_xai_dict_with_no_matching_keys_uses_composite(self) -> None:
        """Legacy branched on the dict being non-empty, not on any candidate
        actually having a score. At w=1.0 that is observable: every composite is
        0.0, so the first candidate wins rather than the best-scoring one."""
        results = {"aug_a": {"accuracy": 0.80}, "aug_b": {"accuracy": 0.90}}
        baseline = {"accuracy": 0.50}
        config = _config(xai_selection_weight=1.0)
        stranger = {"not_a_candidate": 0.7}
        assert select_best_path(results, baseline, config, stranger) == _legacy_select_best_path(
            results, baseline, config, stranger
        )


class TestSelectionResult:
    def test_selected_is_a_tuple(self) -> None:
        result = run_selector(
            {"aug_a": {"accuracy": 0.9}}, {"accuracy": 0.5}, _config()
        )
        assert isinstance(result.selected, tuple)
        assert result.selected == ("aug_a",)

    def test_best_is_none_when_nothing_selected(self) -> None:
        result = run_selector(
            {"aug_a": {"accuracy": 0.4}}, {"accuracy": 0.5}, _config()
        )
        assert result.selected == ()
        assert result.best is None

    @pytest.mark.parametrize(
        ("results", "baseline", "reason"),
        [
            ({"aug_a": {"accuracy": 0.9}}, {"accuracy": 0.5}, "improved"),
            ({"aug_a": {"accuracy": 0.4}}, {"accuracy": 0.5}, "no_improvement"),
            ({}, {"accuracy": 0.5}, "no_candidates"),
            ({"aug_a": {"accuracy": 0.9}}, {}, "no_baseline"),
        ],
    )
    def test_reason_is_a_slug_describing_the_outcome(self, results, baseline, reason) -> None:
        assert run_selector(results, baseline, _config()).reason == reason

    def test_scores_carry_what_was_ranked(self) -> None:
        result = run_selector(
            {"aug_a": {"accuracy": 0.9}, "aug_b": {"accuracy": 0.8}},
            {"accuracy": 0.5},
            _config(),
        )
        assert result.scores == {"aug_a": 0.9, "aug_b": 0.8}

    def test_selector_name_is_recorded(self) -> None:
        result = run_selector({"aug_a": {"accuracy": 0.9}}, {"accuracy": 0.5}, _config())
        assert result.selector == "metric_argmax"


class TestCandidateReport:
    def test_value_reads_the_named_metric(self) -> None:
        report = CandidateReport("aug_a", {"accuracy": 0.9, "f1_macro": 0.7})
        assert report.value("f1_macro") == pytest.approx(0.7)

    def test_missing_metric_is_none_not_zero(self) -> None:
        assert CandidateReport("aug_a", {}).value("accuracy") is None

    def test_xai_score_defaults_to_none(self) -> None:
        assert CandidateReport("aug_a", {}).xai_score is None


class TestRandomSelector:
    def test_switching_selector_changes_the_choice(self) -> None:
        """The point of the registry: the same call, a different rule."""
        results = {f"aug_{i}": {"accuracy": 0.80 + i * 0.01} for i in range(5)}
        baseline = {"accuracy": 0.5}

        argmax = run_selector(results, baseline, _config()).best
        assert argmax == "aug_4"  # the highest accuracy

        picks = {
            run_selector(results, baseline, _config(selector="random", seed=s)).best
            for s in range(30)
        }
        assert len(picks) > 1  # not always the argmax winner
        assert picks <= set(results)

    def test_is_reproducible_for_a_seed(self) -> None:
        results = {f"aug_{i}": {"accuracy": 0.8} for i in range(5)}
        baseline = {"accuracy": 0.5}
        config = _config(selector="random", seed=7)
        first = run_selector(results, baseline, config).best
        second = run_selector(results, baseline, config).best
        assert first == second

    def test_seed_is_not_python_hash_dependent(self) -> None:
        """A hash()-derived seed would differ between processes, which defeats
        the point of seeding at all. Assert against a value computed from the
        stable digest instead."""
        import hashlib

        names = sorted(f"aug_{i}" for i in range(5))
        digest = hashlib.sha256("\x00".join(["7", *names]).encode()).digest()
        expected_rng = random.Random(int.from_bytes(digest[:8], "big"))
        expected = expected_rng.choice(names)

        results = {name: {"accuracy": 0.8} for name in names}
        got = run_selector(results, {"accuracy": 0.5}, _config(selector="random", seed=7)).best
        assert got == expected

    def test_still_gated_on_beating_the_baseline(self) -> None:
        """A random arm that skipped the gate would look better than argmax by
        accepting runs argmax rejects."""
        results = {f"aug_{i}": {"accuracy": 0.4} for i in range(5)}
        result = run_selector(results, {"accuracy": 0.9}, _config(selector="random"))
        assert result.selected == ()
        assert result.reason == "no_improvement"

    def test_select_best_path_honours_the_configured_selector(self) -> None:
        """The legacy entry point routes through the registry too."""
        results = {f"aug_{i}": {"accuracy": 0.80 + i * 0.01} for i in range(5)}
        baseline = {"accuracy": 0.5}
        seeds_giving_non_argmax = [
            s
            for s in range(30)
            if select_best_path(results, baseline, _config(selector="random", seed=s)) != "aug_4"
        ]
        assert seeds_giving_non_argmax


class TestRegistry:
    def test_default_selector_is_metric_argmax(self) -> None:
        assert BNNRConfig().selector == "metric_argmax"

    def test_both_selectors_are_registered(self) -> None:
        assert set(SELECTORS) == {"metric_argmax", "random"}

    def test_unknown_selector_rejected_by_config(self) -> None:
        with pytest.raises(ValueError, match="selector must be one of"):
            BNNRConfig(selector="vibes")

    def test_unknown_selector_rejected_by_lookup(self) -> None:
        with pytest.raises(ValueError, match="Unknown selector"):
            get_selector("vibes")

    def test_every_registered_selector_satisfies_the_protocol(self) -> None:
        for name, selector in SELECTORS.items():
            assert selector.name == name
            result = selector.select(
                [CandidateReport("aug_a", {"accuracy": 0.9})],
                {"accuracy": 0.5},
                _config(),
            )
            assert isinstance(result, SelectionResult)
            assert isinstance(result.selected, tuple)
