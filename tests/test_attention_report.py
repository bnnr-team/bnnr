"""Tests for the attention block in the run report (FIX-3-3)."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.config_model import BNNRConfig
from bnnr.reporting import Reporter
from bnnr.trainer import BNNRTrainer


def _run(tmp_path, tag: str, **config_kwargs):
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
    images = torch.rand(8, 3, 16, 16)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    config = BNNRConfig(
        m_epochs=1,
        max_iterations=0,
        report_dir=tmp_path / tag,
        verbose=False,
        save_checkpoints=False,
        xai_samples=2,
        **config_kwargs,
    )
    trainer = BNNRTrainer(
        model=adapter,
        train_loader=loader,
        val_loader=loader,
        augmentations=[],
        config=config,
        reporter=Reporter(tmp_path / tag, save_html=False),
    )
    return trainer.run()


class TestAttentionBlock:
    def test_the_report_carries_an_attention_block(self, tmp_path) -> None:
        result = _run(tmp_path, "on", xai_enabled=True)
        payload = json.loads(result.report_json_path.read_text())
        assert "attention" in payload["analysis"]

    def test_it_says_which_axis_each_number_is_better_on(self, tmp_path) -> None:
        """Accuracy and calibration reverse the ranking, so a report that prints
        both without saying so invites the reader to assume they agree."""
        result = _run(tmp_path, "on", xai_enabled=True)
        axes = json.loads(result.report_json_path.read_text())["analysis"]["attention"]["axes"]
        assert axes["robustness_gap"] == "lower is better"
        assert axes["ece"] == "lower is better"
        assert axes["accuracy"] == "higher is better"

    def test_it_reports_the_statistics_it_observed(self, tmp_path) -> None:
        result = _run(tmp_path, "on", xai_enabled=True)
        block = json.loads(result.report_json_path.read_text())["analysis"]["attention"]
        assert block["shadow_records"] > 0
        assert block["latest"]["stats"] is not None

    def test_it_names_no_regime_without_thresholds(self, tmp_path) -> None:
        """Naming one needs calibrated thresholds. Their absence is the design,
        so the report says why rather than leaving a silent null."""
        result = _run(tmp_path, "on", xai_enabled=True)
        block = json.loads(result.report_json_path.read_text())["analysis"]["attention"]
        assert block["diagnosis"] is None
        assert "no calibrated thresholds" in block["diagnosis_unavailable_because"]
        assert "docs/diagnosis.md" in block["diagnosis_unavailable_because"]

    def test_no_block_when_there_is_nothing_to_say(self, tmp_path) -> None:
        """With XAI off there are no maps, so an empty block would be noise."""
        result = _run(tmp_path, "off", xai_enabled=False)
        payload = json.loads(result.report_json_path.read_text())
        assert "attention" not in payload["analysis"]

    def test_shadow_mode_off_also_produces_nothing(self, tmp_path) -> None:
        result = _run(tmp_path, "noshadow", xai_enabled=True, shadow_mode=False)
        payload = json.loads(result.report_json_path.read_text())
        assert "attention" not in payload["analysis"]

    def test_the_block_is_json_serialisable(self, tmp_path) -> None:
        """It travels in report.json, so nothing in it may be a numpy scalar."""
        result = _run(tmp_path, "on", xai_enabled=True)
        block = json.loads(result.report_json_path.read_text())["analysis"]["attention"]
        json.dumps(block)  # must not raise

    def test_the_latest_record_names_its_candidate(self, tmp_path) -> None:
        result = _run(tmp_path, "on", xai_enabled=True)
        block = json.loads(result.report_json_path.read_text())["analysis"]["attention"]
        assert block["latest"]["candidate"] == "baseline"


class TestSummarizerAxisLabels:
    """Where a ranking is printed, the axis it ranks on is printed too."""

    def _rows(self):
        return [
            {
                "label": "cond_a", "median": 0.34, "iqr": 0.01, "mean": 0.34,
                "std": 0.01, "n": 10, "delta": "—", "p_holm": "p=0.02",
                "r": "1.00", "ci": "[0, 0]", "gpu_epochs": "40",
                "deployed_epochs": "13", "ece": "0.257",
                "per_seed": "34.00%",
            }
        ]

    def test_the_text_table_states_the_ranking_axis(self, capsys) -> None:
        from benchmarks.summarize_grand import _print_text_table

        _print_text_table(self._rows())
        out = capsys.readouterr().out
        assert "ranked by" in out
        assert "LOWER better" in out

    def test_the_text_table_prints_ece(self, capsys) -> None:
        from benchmarks.summarize_grand import _print_text_table

        _print_text_table(self._rows())
        out = capsys.readouterr().out
        assert "ECE" in out
        assert "0.257" in out

    def test_the_markdown_table_prints_ece_and_its_direction(self, capsys) -> None:
        from benchmarks.summarize_grand import _print_markdown_table

        _print_markdown_table(self._rows())
        out = capsys.readouterr().out
        assert "ECE" in out
        assert "lower is better" in out
        assert "0.257" in out

    def test_a_row_without_ece_prints_a_question_mark(self, capsys) -> None:
        """Records written before ECE was surfaced still summarize."""
        from benchmarks.summarize_grand import _print_text_table

        rows = self._rows()
        rows[0]["ece"] = "?"
        _print_text_table(rows)
        assert "?" in capsys.readouterr().out


class TestEceIsCollectedFromRows:
    def test_the_summarizer_reads_test_ece(self) -> None:
        """The key was recorded from the start and never printed."""
        import json as _json
        from pathlib import Path

        path = Path("benchmarks/results_imagewoof_scratch.json")
        if not path.exists():
            pytest.skip("benchmark results not present")
        data = _json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("runs", [])
        assert any(r.get("test_ece") is not None for r in rows)
