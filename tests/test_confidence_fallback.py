"""Tests for the low-confidence fallback (FIX-4-2)."""

from __future__ import annotations

import pytest

from bnnr.analysis.diagnosis import AttentionRegime, Diagnosis
from bnnr.analysis.saliency_stats import SaliencyStats
from bnnr.config_model import BNNRConfig
from bnnr.training.search_policy import FALLBACK_POLICY, plan_search
from bnnr.training.selection import FALLBACK_SELECTOR, run_selector

CALIBRATED = {
    "concentration_lo": 0.30,
    "concentration_hi": 0.60,
    "border_mass_hi": 0.35,
    "perturbation_shift_hi": 0.50,
    "robustness_gap_hi": 0.15,
}

RESULTS = {"icd": {"accuracy": 0.81}, "aicd": {"accuracy": 0.90}}
BASELINE = {"accuracy": 0.50}
CANDIDATES = ("icd", "aicd")


def _config(*, min_confidence: float | None, **kw) -> BNNRConfig:
    diagnosis = dict(CALIBRATED)
    if min_confidence is not None:
        diagnosis["min_confidence"] = min_confidence
    return BNNRConfig(diagnosis=diagnosis, **kw)


def _diagnosis(confidence: float, recommended: tuple[str, ...] = ("icd",)) -> Diagnosis:
    return Diagnosis(
        regime=AttentionRegime.SHORTCUT_SUSPECTED,
        stats=SaliencyStats(0.5, 0.5, 0.2, (14, 14)),
        overall_acc=0.9,
        hard_quantile_acc=0.4,
        robustness_gap=0.5,
        recommended=recommended,
        confidence=confidence,
        reason="synthetic",
    )


class TestSelectorFallback:
    def test_low_confidence_falls_back_to_metric_argmax(self) -> None:
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=0.75, selector="diagnosis"),
            diagnosis=_diagnosis(0.50),
        )
        assert result.selector == FALLBACK_SELECTOR
        assert result.fallback_from == "diagnosis"
        # metric_argmax picks the better metric, which the diagnosis did not want.
        assert result.best == "aicd"

    def test_sufficient_confidence_keeps_the_diagnosis(self) -> None:
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=0.75, selector="diagnosis"),
            diagnosis=_diagnosis(1.0),
        )
        assert result.selector == "diagnosis"
        assert result.fallback_from is None
        assert result.best == "icd"

    def test_the_threshold_is_inclusive(self) -> None:
        """Exactly at min_confidence counts as confident enough."""
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=0.75, selector="diagnosis"),
            diagnosis=_diagnosis(0.75),
        )
        assert result.fallback_from is None

    def test_the_triggering_confidence_is_recorded(self) -> None:
        """So the threshold can be re-examined without re-running anything."""
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=0.75, selector="diagnosis"),
            diagnosis=_diagnosis(0.25),
        )
        assert result.fallback_confidence == pytest.approx(0.25)

    def test_no_min_confidence_means_no_fallback(self) -> None:
        """The caller has not said where the line is, and inventing one would be
        another uncalibrated number."""
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=None, selector="diagnosis"),
            diagnosis=_diagnosis(0.0),
        )
        assert result.fallback_from is None
        assert result.selector == "diagnosis"

    def test_a_metric_selector_is_never_second_guessed(self) -> None:
        result = run_selector(
            RESULTS, BASELINE,
            _config(min_confidence=0.99, selector="metric_argmax"),
            diagnosis=_diagnosis(0.0),
        )
        assert result.fallback_from is None

    def test_no_diagnosis_is_not_a_fallback(self) -> None:
        """Without one there is nothing to be unconfident about; the selector's
        own no_diagnosis refusal stands."""
        result = run_selector(
            RESULTS, BASELINE, _config(min_confidence=0.75, selector="diagnosis")
        )
        assert result.fallback_from is None
        assert result.reason == "no_diagnosis"


class TestPolicyFallback:
    def test_low_confidence_falls_back_to_exhaustive(self) -> None:
        plan = plan_search(
            CANDIDATES,
            _config(min_confidence=0.75, search_policy="diagnosis_single"),
            diagnosis=_diagnosis(0.50),
        )
        assert plan.policy == FALLBACK_POLICY
        assert plan.fallback_from == "diagnosis_single"

    def test_the_fallback_plan_is_a_real_exhaustive_plan(self) -> None:
        """Not a marker: it must actually evaluate every candidate."""
        plan = plan_search(
            CANDIDATES,
            _config(min_confidence=0.75, search_policy="diagnosis_single", m_epochs=4),
            diagnosis=_diagnosis(0.25),
        )
        assert plan.rungs[0].candidates == CANDIDATES
        assert plan.rungs[0].epochs == 4

    def test_sufficient_confidence_keeps_the_policy(self) -> None:
        plan = plan_search(
            CANDIDATES,
            _config(min_confidence=0.75, search_policy="diagnosis_single"),
            diagnosis=_diagnosis(1.0),
        )
        assert plan.policy == "diagnosis_single"
        assert plan.fallback_from is None

    def test_the_confidence_is_recorded_in_the_plan(self) -> None:
        plan = plan_search(
            CANDIDATES,
            _config(min_confidence=0.75, search_policy="diagnosis_single"),
            diagnosis=_diagnosis(0.5),
        )
        assert plan.to_dict()["fallback_confidence"] == pytest.approx(0.5)
        assert plan.to_dict()["fallback_from"] == "diagnosis_single"

    def test_no_min_confidence_lets_the_policy_run(self) -> None:
        plan = plan_search(
            CANDIDATES,
            _config(min_confidence=None, search_policy="diagnosis_single"),
            diagnosis=_diagnosis(0.0),
        )
        assert plan.policy == "diagnosis_single"

    def test_exhaustive_is_never_second_guessed(self) -> None:
        plan = plan_search(
            CANDIDATES, _config(min_confidence=0.99), diagnosis=_diagnosis(0.0)
        )
        assert plan.fallback_from is None


class TestItReachesTheRunRecord:
    def _run(self, tmp_path, tag: str, **config_kwargs):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from bnnr.adapter import SimpleTorchAdapter
        from bnnr.augmentations import BasicAugmentation, ChurchNoise
        from bnnr.reporting import Reporter
        from bnnr.trainer import BNNRTrainer

        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2),
        )
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device="cpu",
        )
        images = torch.rand(8, 3, 8, 8)
        labels = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        config = BNNRConfig(
            m_epochs=1, max_iterations=1, report_dir=tmp_path / tag,
            verbose=False, save_checkpoints=False, xai_enabled=False,
            **config_kwargs,
        )
        trainer = BNNRTrainer(
            model=adapter, train_loader=loader, val_loader=loader,
            augmentations=[
                BasicAugmentation(probability=1.0, random_state=0),
                ChurchNoise(probability=1.0, random_state=0),
            ],
            config=config,
            reporter=Reporter(tmp_path / tag, save_html=False),
        )
        return trainer.run()

    def test_a_run_without_a_fallback_records_none(self, tmp_path) -> None:
        record = self._run(tmp_path, "plain").run_record
        assert record.selection_fallback_from is None
        assert record.selection_fallback_confidence is None

    def test_the_fields_survive_the_json_round_trip(self, tmp_path) -> None:
        import json

        from bnnr.training.run_record import RunRecord

        result = self._run(tmp_path, "json")
        payload = json.loads(result.report_json_path.read_text())["run_record"]
        assert "selection_fallback_from" in payload
        assert RunRecord.from_dict(payload).selection_fallback_from is None


class TestFallbackRateIsSummarizable:
    """Fallback frequency is itself a calibration signal."""

    def _rows(self, n_fell_back: int, n_eligible: int) -> list[dict]:
        rows = []
        for i in range(n_eligible):
            row = {"dataset": "imagewoof", "selector": "diagnosis", "condition": f"c{i}"}
            if i < n_fell_back:
                row["selection_fallback_from"] = "diagnosis"
                row["selection_fallback_confidence"] = 0.5
            rows.append(row)
        return rows

    def test_it_prints_the_rate(self, capsys) -> None:
        from benchmarks.summarize_grand import _diagnosis_fallback_section

        _diagnosis_fallback_section(self._rows(3, 10), ["imagewoof"])
        out = capsys.readouterr().out
        assert "3/10" in out
        assert "30%" in out

    def test_it_reports_the_median_triggering_confidence(self, capsys) -> None:
        from benchmarks.summarize_grand import _diagnosis_fallback_section

        _diagnosis_fallback_section(self._rows(2, 5), ["imagewoof"])
        assert "0.50" in capsys.readouterr().out

    def test_runs_that_never_asked_for_a_diagnosis_are_not_the_denominator(
        self, capsys
    ) -> None:
        """They are 'not eligible', not 'did not fall back'. Conflating them
        would understate the rate."""
        from benchmarks.summarize_grand import _diagnosis_fallback_section

        rows = self._rows(1, 2) + [
            {"dataset": "imagewoof", "selector": "metric_argmax", "condition": "x"}
        ] * 8
        _diagnosis_fallback_section(rows, ["imagewoof"])
        assert "1/2" in capsys.readouterr().out

    def test_nothing_eligible_prints_nothing(self, capsys) -> None:
        from benchmarks.summarize_grand import _diagnosis_fallback_section

        _diagnosis_fallback_section(
            [{"dataset": "imagewoof", "selector": "metric_argmax"}], ["imagewoof"]
        )
        assert capsys.readouterr().out == ""

    def test_records_from_before_this_change_are_not_counted(self, capsys) -> None:
        from benchmarks.summarize_grand import _diagnosis_fallback_section

        _diagnosis_fallback_section([{"dataset": "imagewoof", "condition": "old"}], ["imagewoof"])
        assert capsys.readouterr().out == ""
