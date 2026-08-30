"""Tests for shadow mode (FIX-3-2)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bnnr.analysis.saliency_stats import SaliencyStats
from bnnr.config_model import BNNRConfig
from bnnr.training.shadow import (
    SHADOW_RECORDS_FILENAME,
    ShadowRecord,
    ShadowRecorder,
    stats_from_maps,
)


def _stats(n_maps: int = 4) -> SaliencyStats:
    return SaliencyStats(
        concentration=0.42,
        gini=0.55,
        border_mass=0.18,
        resolution=(14, 14),
        n_maps=n_maps,
    )


class TestRecorder:
    def test_records_the_raw_statistics_not_a_regime(self) -> None:
        """A regime would need thresholds, and thresholds are what this
        collects evidence for."""
        recorder = ShadowRecorder()
        entry = recorder.record(
            phase="baseline", iteration=0, candidate="baseline",
            stats=_stats(), metrics={"accuracy": 0.9},
        )
        assert "regime" not in entry.to_dict()
        assert entry.stats is not None
        assert entry.stats["concentration"] == pytest.approx(0.42)

    def test_reads_the_robustness_metrics(self) -> None:
        recorder = ShadowRecorder()
        entry = recorder.record(
            phase="candidate", iteration=1, candidate="icd", stats=_stats(),
            metrics={"accuracy": 0.9, "hard_quantile_acc": 0.4, "robustness_gap": 0.5},
        )
        assert entry.overall_acc == pytest.approx(0.9)
        assert entry.hard_quantile_acc == pytest.approx(0.4)
        assert entry.robustness_gap == pytest.approx(0.5)

    def test_absent_metrics_are_none_not_zero(self) -> None:
        """No measurement and a measurement of zero are different facts."""
        recorder = ShadowRecorder()
        entry = recorder.record(
            phase="candidate", iteration=1, candidate="icd",
            stats=_stats(), metrics={"accuracy": 0.9},
        )
        assert entry.hard_quantile_acc is None
        assert entry.robustness_gap is None

    def test_sample_size_travels_with_the_statistics(self) -> None:
        recorder = ShadowRecorder()
        entry = recorder.record(
            phase="baseline", iteration=0, candidate="baseline",
            stats=_stats(n_maps=7), metrics={},
        )
        assert entry.sample_size == 7

    def test_no_stats_records_a_zero_sample(self) -> None:
        recorder = ShadowRecorder()
        entry = recorder.record(
            phase="baseline", iteration=0, candidate="baseline", stats=None, metrics={},
        )
        assert entry.stats is None
        assert entry.sample_size == 0


class TestSelectionFlag:
    def _recorder(self) -> ShadowRecorder:
        recorder = ShadowRecorder()
        for name in ("icd", "aicd", "church_noise"):
            recorder.record(
                phase="candidate", iteration=1, candidate=name, stats=_stats(), metrics={},
            )
        return recorder

    def test_only_the_winner_is_flagged(self) -> None:
        recorder = self._recorder()
        recorder.mark_selected(1, "aicd")
        flagged = [r.candidate for r in recorder.records if r.selected]
        assert flagged == ["aicd"]

    def test_every_candidate_is_kept_not_only_the_winner(self) -> None:
        """A calibration set with only winning arms cannot answer 'would the
        other choice have been better'."""
        recorder = self._recorder()
        recorder.mark_selected(1, "aicd")
        assert len(recorder.records) == 3

    def test_no_selection_flags_nothing(self) -> None:
        recorder = self._recorder()
        recorder.mark_selected(1, None)
        assert not any(r.selected for r in recorder.records)

    def test_a_different_iteration_is_not_touched(self) -> None:
        recorder = self._recorder()
        recorder.record(phase="candidate", iteration=2, candidate="icd", stats=_stats(), metrics={})
        recorder.mark_selected(1, "icd")
        by_iteration = {(r.iteration, r.selected) for r in recorder.records if r.candidate == "icd"}
        assert by_iteration == {(1, True), (2, False)}


class TestWriting:
    def test_writes_one_json_object_per_line(self, tmp_path) -> None:
        recorder = ShadowRecorder()
        for i in range(3):
            recorder.record(
                phase="candidate", iteration=1, candidate=f"aug_{i}",
                stats=_stats(), metrics={"accuracy": 0.8},
            )
        path = recorder.write(tmp_path)
        assert path is not None
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["candidate"] == "aug_0"

    def test_the_filename_is_stable(self, tmp_path) -> None:
        recorder = ShadowRecorder()
        recorder.record(phase="baseline", iteration=0, candidate="baseline", stats=_stats(), metrics={})
        assert recorder.write(tmp_path).name == SHADOW_RECORDS_FILENAME

    def test_nothing_recorded_writes_no_file(self, tmp_path) -> None:
        assert ShadowRecorder().write(tmp_path) is None
        assert not (tmp_path / SHADOW_RECORDS_FILENAME).exists()

    def test_creates_the_directory(self, tmp_path) -> None:
        recorder = ShadowRecorder()
        recorder.record(phase="baseline", iteration=0, candidate="baseline", stats=_stats(), metrics={})
        target = tmp_path / "nested" / "run"
        assert recorder.write(target) is not None


class TestStatsFromMaps:
    def test_aggregates_a_batch(self) -> None:
        maps = np.random.default_rng(0).random((5, 14, 14)).astype(np.float32)
        stats = stats_from_maps(maps)
        assert stats is not None
        assert stats.n_maps == 5

    def test_a_single_map_is_accepted(self) -> None:
        stats = stats_from_maps(np.zeros((14, 14), dtype=np.float32))
        assert stats is not None
        assert stats.n_maps == 1

    def test_empty_maps_contribute_nothing(self) -> None:
        """Not a zeroed record: that would be a misleading 'perfectly uniform
        attention' sample rather than an absent one."""
        assert stats_from_maps(np.zeros((0, 14, 14), dtype=np.float32)) is None

    def test_wrong_rank_is_refused(self) -> None:
        assert stats_from_maps(np.zeros((2, 3, 14, 14), dtype=np.float32)) is None

    def test_perturbation_shift_is_not_computed(self) -> None:
        """It needs a second explainer pass, and shadow mode's whole claim is
        that it costs nothing."""
        stats = stats_from_maps(np.random.default_rng(1).random((3, 14, 14)).astype(np.float32))
        assert stats is not None
        assert stats.perturbation_shift is None


class TestConfig:
    def test_shadow_mode_is_on_by_default(self) -> None:
        assert BNNRConfig().shadow_mode is True

    def test_it_can_be_turned_off(self) -> None:
        assert BNNRConfig(shadow_mode=False).shadow_mode is False


class TestSelectionIsUnchanged:
    """The load-bearing claim: shadow mode observes and changes nothing."""

    def _run(self, tmp_path, *, shadow: bool):
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
            m_epochs=1,
            max_iterations=1,
            report_dir=tmp_path / f"reports_{shadow}",
            verbose=False,
            save_checkpoints=False,
            xai_enabled=False,
            shadow_mode=shadow,
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
            reporter=Reporter(tmp_path / f"reports_{shadow}", save_html=False),
        )
        return trainer.run()

    def test_the_same_candidate_is_selected_either_way(self, tmp_path) -> None:
        with_shadow = self._run(tmp_path, shadow=True)
        without = self._run(tmp_path, shadow=False)
        assert with_shadow.best_path == without.best_path
        assert with_shadow.selected_augmentations == without.selected_augmentations

    def test_the_metrics_are_identical(self, tmp_path) -> None:
        with_shadow = self._run(tmp_path, shadow=True)
        without = self._run(tmp_path, shadow=False)
        assert with_shadow.best_metrics == without.best_metrics

    def test_disabled_shadow_mode_writes_no_file(self, tmp_path) -> None:
        result = self._run(tmp_path, shadow=False)
        assert not (result.report_json_path.parent / SHADOW_RECORDS_FILENAME).exists()


class TestRecordShape:
    def test_a_record_serialises_to_json(self) -> None:
        entry = ShadowRecord(phase="baseline", iteration=0, candidate="baseline")
        assert json.loads(json.dumps(entry.to_dict()))["phase"] == "baseline"

    def test_selected_starts_false(self) -> None:
        assert ShadowRecord(phase="candidate", iteration=1, candidate="icd").selected is False


class TestARealRunProducesTheRecordsFile:
    """The 'done when' criterion: a normal run with XAI on writes a record for
    every candidate, and selection is still unchanged."""

    def _run(self, tmp_path, *, shadow: bool, tag: str):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from bnnr.adapter import SimpleTorchAdapter
        from bnnr.augmentations import BasicAugmentation, ChurchNoise
        from bnnr.reporting import Reporter
        from bnnr.trainer import BNNRTrainer

        torch.manual_seed(0)
        conv = nn.Conv2d(3, 4, 3, padding=1)
        model = nn.Sequential(
            conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)
        )
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device="cpu",
        )
        images = torch.rand(8, 3, 16, 16)
        labels = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        config = BNNRConfig(
            m_epochs=1,
            max_iterations=1,
            report_dir=tmp_path / tag,
            verbose=False,
            save_checkpoints=False,
            xai_enabled=True,
            xai_samples=2,
            shadow_mode=shadow,
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

    def _records(self, result) -> list[dict]:
        path = result.report_json_path.parent / SHADOW_RECORDS_FILENAME
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().strip().split("\n")]

    def test_a_record_exists_for_every_candidate(self, tmp_path) -> None:
        result = self._run(tmp_path, shadow=True, tag="on")
        candidates = {r["candidate"] for r in self._records(result) if r["phase"] == "candidate"}
        assert candidates == {"basic_augmentation", "church_noise"}

    def test_the_baseline_phase_is_recorded_too(self, tmp_path) -> None:
        result = self._run(tmp_path, shadow=True, tag="on")
        phases = {r["phase"] for r in self._records(result)}
        assert "baseline" in phases

    def test_records_carry_statistics_and_a_sample_size(self, tmp_path) -> None:
        result = self._run(tmp_path, shadow=True, tag="on")
        records = self._records(result)
        assert records
        for record in records:
            assert record["stats"] is not None
            assert record["stats"]["resolution"] == [14, 14]
            assert record["sample_size"] > 0

    def test_no_record_claims_a_regime(self, tmp_path) -> None:
        result = self._run(tmp_path, shadow=True, tag="on")
        for record in self._records(result):
            assert "regime" not in record
            assert "recommended" not in record

    def test_selection_is_unchanged_with_records_actually_written(
        self, tmp_path
    ) -> None:
        """The claim that matters, tested where shadow mode is doing work."""
        with_shadow = self._run(tmp_path, shadow=True, tag="on")
        without = self._run(tmp_path, shadow=False, tag="off")
        assert self._records(with_shadow)  # it really did record
        assert not self._records(without)
        assert with_shadow.best_path == without.best_path
        assert with_shadow.best_metrics == without.best_metrics
