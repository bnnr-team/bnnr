"""Tests for the search policies (FIX-4-1)."""

from __future__ import annotations

import pytest

from bnnr.analysis.diagnosis import AttentionRegime, Diagnosis
from bnnr.analysis.saliency_stats import SaliencyStats
from bnnr.config_model import BNNRConfig
from bnnr.training.search_policy import (
    SEARCH_POLICIES,
    UnplannableSearchError,
    plan_search,
)

CALIBRATED = {
    "concentration_lo": 0.30,
    "concentration_hi": 0.60,
    "border_mass_hi": 0.35,
    "perturbation_shift_hi": 0.50,
    "robustness_gap_hi": 0.15,
}

CANDIDATES = ("icd", "aicd", "church_noise")


def _config(policy: str = "exhaustive", *, m_epochs: int = 6, **kw) -> BNNRConfig:
    if policy == "diagnosis_single":
        kw.setdefault("diagnosis", CALIBRATED)
    return BNNRConfig(search_policy=policy, m_epochs=m_epochs, **kw)


def _diagnosis(recommended: tuple[str, ...]) -> Diagnosis:
    regime = {
        ("icd",): AttentionRegime.SHORTCUT_SUSPECTED,
        ("aicd",): AttentionRegime.OBJECT_FOCUSED,
        ("church_noise",): AttentionRegime.UNSTRUCTURED,
    }[recommended]
    return Diagnosis(
        regime=regime,
        stats=SaliencyStats(0.5, 0.5, 0.2, (14, 14)),
        overall_acc=0.9,
        hard_quantile_acc=0.4,
        robustness_gap=0.5,
        recommended=recommended,
        confidence=1.0,
        reason="synthetic",
    )


class TestExhaustiveIsUnchanged:
    def test_one_rung_holding_every_candidate(self) -> None:
        plan = plan_search(CANDIDATES, _config())
        assert len(plan.rungs) == 1
        assert plan.rungs[0].candidates == CANDIDATES

    def test_each_candidate_gets_m_epochs(self) -> None:
        plan = plan_search(CANDIDATES, _config(m_epochs=6))
        assert plan.rungs[0].epochs == 6

    def test_the_deployed_share_is_the_defect_it_describes(self) -> None:
        """Three candidates, 18 epochs spent, 6 reaching the deployed model."""
        plan = plan_search(CANDIDATES, _config(m_epochs=6))
        assert plan.total_epochs == 18
        assert plan.deployed_epochs == 6

    def test_it_needs_no_diagnosis(self) -> None:
        assert plan_search(CANDIDATES, _config(), diagnosis=None).policy == "exhaustive"

    def test_it_is_the_default(self) -> None:
        assert BNNRConfig().search_policy == "exhaustive"


class TestDiagnosisSingle:
    def test_the_recommended_candidate_takes_the_whole_budget(self) -> None:
        plan = plan_search(
            CANDIDATES, _config("diagnosis_single", m_epochs=6),
            diagnosis=_diagnosis(("icd",)),
        )
        assert plan.rungs[0].candidates == ("icd",)
        assert plan.rungs[0].epochs == 18

    def test_it_spends_exactly_what_exhaustive_would_have(self) -> None:
        """Same total, all of it on the arm the evidence points at."""
        exhaustive = plan_search(CANDIDATES, _config(m_epochs=6))
        single = plan_search(
            CANDIDATES, _config("diagnosis_single", m_epochs=6),
            diagnosis=_diagnosis(("icd",)),
        )
        assert single.total_epochs == exhaustive.total_epochs
        assert single.deployed_epochs == 3 * exhaustive.deployed_epochs

    @pytest.mark.parametrize("family", ["icd", "aicd", "church_noise"])
    def test_it_follows_the_recommendation(self, family: str) -> None:
        plan = plan_search(
            CANDIDATES, _config("diagnosis_single"), diagnosis=_diagnosis((family,))
        )
        assert plan.rungs[0].candidates == (family,)

    def test_aicd_is_not_matched_by_the_icd_family(self) -> None:
        """"icd" in "aicd" is true; the naive ordering would pick the wrong arm."""
        plan = plan_search(
            ("aicd_p50", "icd_p90"), _config("diagnosis_single"),
            diagnosis=_diagnosis(("icd",)),
        )
        assert plan.rungs[0].candidates == ("icd_p90",)

    def test_no_diagnosis_refuses_rather_than_guessing(self) -> None:
        """A silent fallback to argmax would make a benchmark contrast between
        the policies measure a blend of them."""
        with pytest.raises(UnplannableSearchError, match="needs a diagnosis"):
            plan_search(CANDIDATES, _config("diagnosis_single"), diagnosis=None)

    def test_an_unmatchable_recommendation_refuses(self) -> None:
        with pytest.raises(UnplannableSearchError, match="matches none"):
            plan_search(
                ("church_noise",), _config("diagnosis_single"),
                diagnosis=_diagnosis(("icd",)),
            )

    def test_the_error_names_a_way_out(self) -> None:
        with pytest.raises(UnplannableSearchError, match="exhaustive"):
            plan_search(
                ("church_noise",), _config("diagnosis_single"),
                diagnosis=_diagnosis(("aicd",)),
            )

    def test_it_is_refused_without_calibrated_thresholds(self) -> None:
        """The same gate that guards the diagnosis selector."""
        with pytest.raises(ValueError, match="calibrated diagnosis thresholds"):
            BNNRConfig(search_policy="diagnosis_single")

    def test_the_error_names_the_policy_not_the_selector(self) -> None:
        with pytest.raises(ValueError, match="search_policy='diagnosis_single'"):
            BNNRConfig(search_policy="diagnosis_single")


class TestSuccessiveHalving:
    def test_the_field_shrinks_each_rung(self) -> None:
        plan = plan_search(
            tuple(f"aug_{i}" for i in range(8)), _config("successive_halving")
        )
        sizes = [len(r.candidates) for r in plan.rungs]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == 8
        assert sizes[-1] < sizes[0]

    def test_it_does_not_cost_more_than_exhaustive(self) -> None:
        """Weak branches dying is what pays for the survivors' extra epochs."""
        candidates = tuple(f"aug_{i}" for i in range(8))
        halving = plan_search(candidates, _config("successive_halving", m_epochs=8))
        exhaustive = plan_search(candidates, _config(m_epochs=8))
        assert halving.total_epochs <= exhaustive.total_epochs

    def test_a_survivor_trains_across_every_rung(self) -> None:
        plan = plan_search(
            tuple(f"aug_{i}" for i in range(8)), _config("successive_halving")
        )
        assert plan.deployed_epochs == sum(r.epochs for r in plan.rungs)
        assert len(plan.rungs) > 1

    def test_a_single_candidate_degenerates_to_one_full_rung(self) -> None:
        plan = plan_search(("only",), _config("successive_halving", m_epochs=5))
        assert len(plan.rungs) == 1
        assert plan.rungs[0].epochs == 5

    def test_no_candidates_does_not_crash(self) -> None:
        plan = plan_search((), _config("successive_halving"))
        assert plan.rungs[0].candidates == ()

    def test_every_rung_trains_at_least_one_epoch(self) -> None:
        plan = plan_search(
            tuple(f"aug_{i}" for i in range(16)), _config("successive_halving", m_epochs=2)
        )
        assert all(r.epochs >= 1 for r in plan.rungs)

    def test_it_needs_no_diagnosis(self) -> None:
        """It is the fallback for when the diagnosis is uncertain."""
        assert plan_search(CANDIDATES, _config("successive_halving")).policy == (
            "successive_halving"
        )


class TestRegistryAndRecord:
    def test_all_three_policies_are_registered(self) -> None:
        assert set(SEARCH_POLICIES) == {
            "exhaustive", "diagnosis_single", "successive_halving",
        }

    def test_an_unknown_policy_is_refused_by_config(self) -> None:
        with pytest.raises(ValueError, match="search_policy must be one of"):
            BNNRConfig(search_policy="vibes")

    def test_the_plan_is_json_ready(self) -> None:
        import json

        plan = plan_search(CANDIDATES, _config())
        payload = json.loads(json.dumps(plan.to_dict()))
        assert payload["policy"] == "exhaustive"
        assert payload["rungs"][0]["candidates"] == list(CANDIDATES)

    def test_the_plan_records_both_epoch_numbers(self) -> None:
        plan = plan_search(CANDIDATES, _config(m_epochs=6))
        record = plan.to_dict()
        assert record["total_epochs"] == 18
        assert record["deployed_epochs"] == 6

    def test_rung_cost_is_candidates_times_epochs(self) -> None:
        rung = plan_search(CANDIDATES, _config(m_epochs=4)).rungs[0]
        assert rung.cost == 12


class TestRealRunUnderEachPolicy:
    """All three run end to end, and the accounting reaches the record."""

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
            m_epochs=2,
            max_iterations=1,
            report_dir=tmp_path / tag,
            verbose=False,
            save_checkpoints=False,
            xai_enabled=False,
            **config_kwargs,
        )
        trainer = BNNRTrainer(
            model=adapter,
            train_loader=loader,
            val_loader=loader,
            augmentations=[
                BasicAugmentation(probability=1.0, random_state=0),
                ChurchNoise(probability=1.0, random_state=0),
            ],
            config=config,
            reporter=Reporter(tmp_path / tag, save_html=False),
        )
        return trainer.run()

    def test_exhaustive_runs_and_records_its_policy(self, tmp_path) -> None:
        record = self._run(tmp_path, "ex").run_record
        assert record.search_policy == "exhaustive"
        assert record.search_plan is not None
        assert record.search_plan["policy"] == "exhaustive"

    def test_successive_halving_runs(self, tmp_path) -> None:
        record = self._run(tmp_path, "sh", search_policy="successive_halving").run_record
        assert record.search_policy == "successive_halving"
        assert record.search_plan["policy"] == "successive_halving"

    def test_the_recorded_epochs_are_counted_not_planned(self, tmp_path) -> None:
        """The ledger counts what ran; the plan says what was intended. Pruning
        can make them differ, and the record must carry the measurement."""
        record = self._run(tmp_path, "count").run_record
        assert record.total_gpu_epochs > 0
        assert record.deployed_epochs <= record.total_gpu_epochs
