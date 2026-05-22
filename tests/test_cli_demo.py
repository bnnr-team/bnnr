"""CLI tests for ``bnnr demo``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bnnr.cli import _print_demo_followup, _resolve_xai_artifact_dir, _run_train, app
from bnnr.icd import ICD
from bnnr.reporting import BNNRRunResult

runner = CliRunner()


def _mock_run_result(tmp_path: Path, *, with_xai: bool = True) -> BNNRRunResult:
    run_dir = tmp_path / "reports" / "run_demo"
    run_dir.mkdir(parents=True)
    if with_xai:
        xai_dir = run_dir / "artifacts" / "xai" / "iter_1_icd" / "epoch_1"
        xai_dir.mkdir(parents=True)
        (xai_dir / "heatmap.png").write_bytes(b"png")
    report_json = run_dir / "report.json"
    report_json.write_text("{}", encoding="utf-8")
    from bnnr.config_model import BNNRConfig

    return BNNRRunResult(
        config=BNNRConfig(),
        checkpoints=[],
        best_path="baseline -> icd",
        best_metrics={"accuracy": 0.5},
        selected_augmentations=["icd"],
        total_time=1.0,
        report_json_path=report_json,
        report_html_path=None,
    )


class TestDemoCommand:
    def test_demo_help_lists_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "demo" in result.stdout

    def test_demo_invokes_train_with_on_complete(self, tmp_path: Path) -> None:
        mock_result = _mock_run_result(tmp_path)

        with patch("bnnr.cli._run_train", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["demo"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["dataset"] == "cifar10"
        assert call_kwargs["augmentation_preset"] == "demo"
        assert call_kwargs["max_train_samples"] == 128
        assert call_kwargs["on_complete"] is _print_demo_followup

    def test_demo_followup_messages_with_artifacts_xai(self, tmp_path: Path) -> None:
        mock_result = _mock_run_result(tmp_path)
        with patch("bnnr.cli.typer.echo") as echo:
            _print_demo_followup(mock_result)
        combined = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert "Your report:" in combined
        assert "artifacts/xai" in combined.replace("\\", "/")
        assert "XAI heatmaps" in combined

    def test_resolve_xai_artifact_dir_prefers_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        legacy = run_dir / "xai"
        legacy.mkdir(parents=True)
        (legacy / "old.png").write_bytes(b"x")
        canonical = run_dir / "artifacts" / "xai" / "iter_0"
        canonical.mkdir(parents=True)
        (canonical / "new.png").write_bytes(b"x")

        resolved = _resolve_xai_artifact_dir(run_dir)
        assert resolved == run_dir / "artifacts" / "xai"


class TestRunTrainOnCompleteOrdering:
    def test_on_complete_runs_before_dashboard_wait(self, tmp_path: Path) -> None:
        """Demo follow-up must print while training is done, not after Ctrl+C."""
        mock_result = _mock_run_result(tmp_path, with_xai=False)
        order: list[str] = []

        def on_complete(_result: object) -> None:
            order.append("on_complete")

        cfg = MagicMock()
        cfg.report_dir = tmp_path / "reports"
        cfg.seed = 42

        with (
            patch(
                "bnnr.pipelines.build_pipeline",
                return_value=(MagicMock(), MagicMock(), MagicMock(), []),
            ),
            patch("bnnr.cli._print_pipeline_summary"),
            patch("bnnr.cli.start_dashboard", return_value="http://127.0.0.1:8080/"),
            patch("bnnr.cli.BNNRTrainer") as trainer_cls,
            patch("bnnr.cli.time.sleep", side_effect=InterruptedError),
        ):
            trainer_cls.return_value.run.return_value = mock_result
            try:
                _run_train(
                    cfg=cfg,
                    dataset="cifar10",
                    data_dir=tmp_path,
                    data_path=None,
                    augmentation_preset="demo",
                    with_dashboard=True,
                    dashboard_port=8080,
                    no_auto_open=True,
                    dashboard_token=None,
                    batch_size=64,
                    max_train_samples=128,
                    max_val_samples=64,
                    num_classes=None,
                    on_complete=on_complete,
                )
            except InterruptedError:
                order.append("dashboard_wait")

        assert order == ["on_complete", "dashboard_wait"]


class TestDemoPresetIntegration:
    def test_demo_preset_icd_requires_model(self) -> None:
        import torch.nn as nn

        from bnnr.presets import get_preset

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
        augs = get_preset("demo", model=model, target_layers=[model[0]])
        assert any(isinstance(a, ICD) for a in augs)
