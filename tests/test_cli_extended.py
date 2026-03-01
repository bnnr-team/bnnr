"""Extended CLI tests covering list commands and report formatting.

These tests exercise the CLI commands via ``typer.testing.CliRunner``
to avoid actually downloading datasets or starting servers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from bnnr.cli import app
from bnnr.core import BNNRConfig

runner = CliRunner()


class TestVersionCommand:
    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "bnnr version" in result.stdout


class TestListDatasetsCommand:
    def test_lists_datasets(self):
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0
        assert "mnist" in result.stdout
        assert "cifar10" in result.stdout
        assert "imagefolder" in result.stdout


class TestListPresetsCommand:
    def test_lists_presets(self):
        result = runner.invoke(app, ["list-presets"])
        assert result.exit_code == 0
        assert "auto" in result.stdout
        assert "light" in result.stdout
        assert "standard" in result.stdout


class TestListAugmentationsCommand:
    def test_lists_augmentations(self):
        result = runner.invoke(app, ["list-augmentations"])
        assert result.exit_code == 0
        # Should list at least one augmentation
        assert len(result.stdout.strip()) > 0

    def test_verbose_flag(self):
        result = runner.invoke(app, ["list-augmentations", "--verbose"])
        assert result.exit_code == 0
        # Verbose output should include "class:" or "doc:" info
        assert "class:" in result.stdout.lower() or "doc:" in result.stdout.lower()


class TestReportCommand:
    """Tests for 'bnnr report' with mock report data."""

    def _create_mock_report(self, tmp_path: Path) -> Path:
        """Create a minimal BNNR report JSON file matching load_report format."""
        report = {
            "config": BNNRConfig().model_dump(mode="json"),
            "best_path": "baseline -> ChurchNoise",
            "best_metrics": {"accuracy": 0.95},
            "selected_augmentations": ["ChurchNoise"],
            "total_time": 42.0,
            "checkpoints": [
                {
                    "iteration": 1,
                    "augmentation": "ChurchNoise",
                    "epoch": 5,
                    "metrics": {"accuracy": 0.95},
                    "checkpoint_path": "checkpoints/cp.pt",
                    "xai_paths": [],
                    "preview_pairs": [],
                    "probe_labels": [],
                    "active_path": "baseline -> ChurchNoise",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "analysis": {},
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        return path

    def test_summary_format(self, tmp_path):
        report_path = self._create_mock_report(tmp_path)
        result = runner.invoke(app, ["report", str(report_path)])
        assert result.exit_code == 0
        assert "Best path" in result.stdout
        assert "ChurchNoise" in result.stdout

    def test_json_format(self, tmp_path):
        report_path = self._create_mock_report(tmp_path)
        result = runner.invoke(app, ["report", str(report_path), "-f", "json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["best_path"] == "baseline -> ChurchNoise"
        assert "best_metrics" in data

    def test_html_format_removed(self, tmp_path):
        report_path = self._create_mock_report(tmp_path)
        result = runner.invoke(app, ["report", str(report_path), "-f", "html"])
        assert result.exit_code == 1

    def test_invalid_format(self, tmp_path):
        report_path = self._create_mock_report(tmp_path)
        result = runner.invoke(app, ["report", str(report_path), "-f", "pdf"])
        assert result.exit_code == 1


class TestXaiPresets:
    """Tests for XAI preset helpers in config.py."""

    def test_list_xai_presets(self):
        from bnnr.config import list_xai_presets

        presets = list_xai_presets()
        assert isinstance(presets, list)
        assert "xai_light" in presets
        assert "xai_full" in presets
        assert "xai_adaptive" in presets

    def test_get_xai_preset_valid(self):
        from bnnr.config import get_xai_preset

        preset = get_xai_preset("xai_light")
        assert isinstance(preset, dict)
        assert "xai_enabled" in preset
        assert preset["xai_enabled"] is True

    def test_get_xai_preset_invalid(self):
        from bnnr.config import get_xai_preset

        with pytest.raises(KeyError, match="Unknown XAI preset"):
            get_xai_preset("nonexistent_preset")

    def test_apply_xai_preset(self):
        from bnnr.config import apply_xai_preset

        cfg = BNNRConfig(xai_enabled=False)
        new_cfg = apply_xai_preset(cfg, "xai_full")
        assert new_cfg.xai_enabled is True
        assert new_cfg.dual_xai_report is True
        assert new_cfg.xai_selection_weight == 0.1

    def test_apply_xai_preset_preserves_other_fields(self):
        from bnnr.config import apply_xai_preset

        cfg = BNNRConfig(m_epochs=99, seed=777)
        new_cfg = apply_xai_preset(cfg, "xai_adaptive")
        assert new_cfg.m_epochs == 99
        assert new_cfg.seed == 777
        assert new_cfg.xai_enabled is True


class TestPrintPipelineSummary:
    """Test _print_pipeline_summary to hit its various branches."""

    def test_non_imagefolder_prints_demo_note(self):
        from bnnr.cli import _print_pipeline_summary

        class _FakeModel(nn.Module):
            def forward(self, x):
                return x

        class _FakeAdapter:
            def __init__(self):
                self.optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
                self.scheduler = None

            def get_model(self):
                return _FakeModel()

        class _FakeLoader:
            batch_size = 32

            class Dataset:
                def __len__(self):
                    return 100

            dataset = Dataset()

        _print_pipeline_summary(
            dataset_name="mnist",
            adapter=_FakeAdapter(),
            train_loader=_FakeLoader(),
            val_loader=_FakeLoader(),
            augmentations=[],
            config=BNNRConfig(),
            preset="auto",
        )

    def test_imagefolder_prints_generic_note(self):
        from bnnr.cli import _print_pipeline_summary

        class _FakeModel(nn.Module):
            def forward(self, x):
                return x

        class _FakeAdapter:
            optimizer = None
            scheduler = None

            def get_model(self):
                return _FakeModel()

        class _FakeLoader:
            batch_size = 16

            class Dataset:
                def __len__(self):
                    raise TypeError("no len")

            dataset = Dataset()

        _print_pipeline_summary(
            dataset_name="imagefolder",
            adapter=_FakeAdapter(),
            train_loader=_FakeLoader(),
            val_loader=_FakeLoader(),
            augmentations=[],
            config=BNNRConfig(),
            preset="light",
            custom_data_path=Path("/fake/path"),
        )
