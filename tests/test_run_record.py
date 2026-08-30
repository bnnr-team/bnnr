"""Tests for the mandatory run-record fields (FIX-3-1)."""

from __future__ import annotations

import json

import pytest

from bnnr.training.run_record import (
    ComputeLedger,
    RunRecord,
    collect_augmentation_modes,
)


class TestComputeLedger:
    def test_counts_every_trained_epoch(self) -> None:
        ledger = ComputeLedger()
        for _ in range(7):
            ledger.count_trained_epoch()
        assert ledger.total_gpu_epochs == 7

    def test_deployed_is_credited_separately(self) -> None:
        """The two are different quantities, not two names for one."""
        ledger = ComputeLedger()
        for _ in range(12):
            ledger.count_trained_epoch()
        ledger.credit_deployed(3)
        ledger.credit_deployed(2)
        assert ledger.total_gpu_epochs == 12
        assert ledger.deployed_epochs == 5

    def test_crediting_zero_is_a_no_op(self) -> None:
        """An iteration that kept nothing must not move the deployed count."""
        ledger = ComputeLedger()
        ledger.credit_deployed(0)
        assert ledger.deployed_epochs == 0

    def test_starts_at_zero(self) -> None:
        ledger = ComputeLedger()
        assert (ledger.total_gpu_epochs, ledger.deployed_epochs) == (0, 0)


class TestRunRecordRoundTrip:
    def _record(self) -> RunRecord:
        return RunRecord(
            total_gpu_epochs=45,
            deployed_epochs=15,
            search_policy="exhaustive",
            selector="metric_argmax",
            selected_candidate=("icd", "church_noise"),
            diagnosis={"regime": "shortcut_suspected", "confidence": 0.75},
            hard_quantile_q=0.2,
            augmentation_modes={"church_noise": "regional"},
        )

    def test_to_dict_is_json_serialisable(self) -> None:
        payload = json.dumps(self._record().to_dict())
        assert "shortcut_suspected" in payload

    def test_selected_candidate_becomes_a_list_in_json(self) -> None:
        assert self._record().to_dict()["selected_candidate"] == ["icd", "church_noise"]

    def test_from_dict_restores_the_tuple(self) -> None:
        restored = RunRecord.from_dict(self._record().to_dict())
        assert restored.selected_candidate == ("icd", "church_noise")
        assert isinstance(restored.selected_candidate, tuple)

    def test_round_trip_preserves_every_field(self) -> None:
        original = self._record()
        assert RunRecord.from_dict(original.to_dict()) == original


class TestOldRecordsStillRead:
    """Records written before this change must summarize, not crash."""

    def test_an_empty_record_falls_back_to_defaults(self) -> None:
        record = RunRecord.from_dict({})
        assert record.total_gpu_epochs == 0
        assert record.search_policy == "exhaustive"

    def test_none_is_tolerated(self) -> None:
        assert RunRecord.from_dict(None) == RunRecord()

    def test_a_record_missing_new_fields_loads(self) -> None:
        """The shape a pre-FIX-3-1 row would have."""
        record = RunRecord.from_dict({"total_gpu_epochs": 30})
        assert record.total_gpu_epochs == 30
        assert record.deployed_epochs == 0
        assert record.diagnosis is None

    def test_an_unknown_key_is_dropped_rather_than_raising(self) -> None:
        """A record written by a later version must not break an older reader."""
        record = RunRecord.from_dict({"total_gpu_epochs": 5, "field_from_the_future": 1})
        assert record.total_gpu_epochs == 5

    def test_a_null_selected_candidate_does_not_crash(self) -> None:
        assert RunRecord.from_dict({"selected_candidate": None}).selected_candidate == ()


class _FakeAug:
    def __init__(self, name: str, **modes: str) -> None:
        self.name = name
        for key, value in modes.items():
            setattr(self, key, value)


class TestAugmentationModes:
    def test_reads_each_mode_attribute(self) -> None:
        augs = [
            _FakeAug("church_noise", noise_mode="regional"),
            _FakeAug("dif_presets", effect_mode="circles"),
            _FakeAug("procam", camera_mode="profile"),
        ]
        assert collect_augmentation_modes(augs) == {
            "church_noise": "regional",
            "dif_presets": "circles",
            "procam": "profile",
        }

    def test_augmentations_without_a_mode_contribute_nothing(self) -> None:
        """The mapping is a statement about what was configurable."""
        assert collect_augmentation_modes([_FakeAug("drust")]) == {}

    def test_empty_list_is_empty(self) -> None:
        assert collect_augmentation_modes([]) == {}


class TestRealRunPopulatesTheRecord:
    """The counts must come from a real run, not from config arithmetic."""

    def _trainer(self, tmp_path, **config_kwargs):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from bnnr.adapter import SimpleTorchAdapter
        from bnnr.config_model import BNNRConfig
        from bnnr.reporting import Reporter
        from bnnr.trainer import BNNRTrainer

        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
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
            max_iterations=0,
            report_dir=tmp_path / "reports",
            verbose=False,
            save_checkpoints=False,
            # A two-layer Linear stack has no conv layer for OptiCAM to hook,
            # and this test is about the epoch counts, not about saliency.
            xai_enabled=False,
            **config_kwargs,
        )
        return BNNRTrainer(
            model=adapter,
            train_loader=loader,
            val_loader=loader,
            augmentations=[],
            config=config,
            reporter=Reporter(tmp_path / "reports", save_html=False),
        )

    def test_a_baseline_only_run_records_both_counts(self, tmp_path) -> None:
        result = self._trainer(tmp_path).run()
        assert result.run_record.total_gpu_epochs == 2
        assert result.run_record.deployed_epochs > 0

    def test_deployed_never_exceeds_total(self, tmp_path) -> None:
        record = self._trainer(tmp_path).run().run_record
        assert record.deployed_epochs <= record.total_gpu_epochs

    def test_the_selector_and_quantile_are_recorded(self, tmp_path) -> None:
        record = self._trainer(tmp_path, hard_quantile_q=0.3).run().run_record
        assert record.selector == "metric_argmax"
        assert record.hard_quantile_q == pytest.approx(0.3)

    def test_search_policy_is_recorded_before_alternatives_exist(self, tmp_path) -> None:
        """So rows from before #413 stay distinguishable from rows after it."""
        assert self._trainer(tmp_path).run().run_record.search_policy == "exhaustive"

    def test_the_record_lands_in_the_json_report(self, tmp_path) -> None:
        result = self._trainer(tmp_path).run()
        payload = json.loads(result.report_json_path.read_text())
        assert "run_record" in payload
        assert payload["run_record"]["total_gpu_epochs"] == 2

    def test_the_json_report_still_loads(self, tmp_path) -> None:
        """A new key must not break the reader that validates required ones."""
        from bnnr.reporting import load_report

        result = self._trainer(tmp_path).run()
        assert load_report(result.report_json_path) is not None
