"""Integration tests across primary BNNR workflows."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typer.testing import CliRunner

from bnnr import cli as cli_module
from bnnr import pipelines as pipelines_module
from bnnr.augmentations import AugmentationRegistry, BasicAugmentation
from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter


class GrayCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).reshape(x.shape[0], -1)
        return self.fc(x)


def _gray_loader() -> DataLoader:
    x = torch.rand(12, 1, 28, 28)
    y = torch.randint(0, 10, (12,))
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def test_e2e_grayscale_training_with_augmentation_alias(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
        save_checkpoints=True,
    )
    aug = AugmentationRegistry.create("augmentation_3", probability=1.0, random_state=7)
    trainer = BNNRTrainer(adapter, loader, loader, [aug], cfg)
    result = trainer.run()

    assert result.report_json_path.exists()
    assert result.report_html_path is None


def test_resume_from_checkpoint_restores_iteration(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
    )
    trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.5)], cfg)
    trainer.run()

    checkpoint_paths = sorted(cfg.checkpoint_dir.glob("*.pt"))
    assert checkpoint_paths, "Checkpoint should be created"

    new_trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.5)], cfg)
    new_trainer.resume_from_checkpoint(checkpoint_paths[-1])
    assert new_trainer.current_iteration >= 0
    assert isinstance(new_trainer._resume_completed_candidates, set)


def test_cli_report_summary_command(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
    )
    trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.3)], cfg)
    result = trainer.run()

    runner = CliRunner()
    cli_result = runner.invoke(cli_module.app, ["report", str(result.report_json_path), "--format", "summary"])
    assert cli_result.exit_code == 0
    assert "Best path" in cli_result.output


def test_cli_report_summary_command_writes_output_file(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
    )
    trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.3)], cfg)
    result = trainer.run()

    out_path = temp_dir / "report_summary.txt"
    runner = CliRunner()
    cli_result = runner.invoke(
        cli_module.app,
        ["report", str(result.report_json_path), "--format", "summary", "--output", str(out_path)],
    )
    assert cli_result.exit_code == 0
    assert "Saved report to:" in cli_result.output
    assert out_path.exists()
    rendered = out_path.read_text(encoding="utf-8")
    assert "Best path:" in rendered
    assert "Total checkpoints:" in rendered


def test_cli_report_json_command_writes_output_file(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
    )
    trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.3)], cfg)
    result = trainer.run()

    out_path = temp_dir / "report_payload.json"
    runner = CliRunner()
    cli_result = runner.invoke(
        cli_module.app,
        ["report", str(result.report_json_path), "--format", "json", "--output", str(out_path)],
    )
    assert cli_result.exit_code == 0
    assert "Saved report to:" in cli_result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "best_path" in payload
    assert "best_metrics" in payload


def test_cli_report_html_is_deprecated(temp_dir) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "checkpoints",
        report_dir=temp_dir / "reports",
    )
    trainer = BNNRTrainer(adapter, loader, loader, [BasicAugmentation(probability=0.3)], cfg)
    result = trainer.run()
    runner = CliRunner()
    cli_result = runner.invoke(cli_module.app, ["report", str(result.report_json_path), "--format", "html"])
    assert cli_result.exit_code == 1
    assert "dashboard export" in cli_result.output


def test_cli_train_command_with_monkeypatched_pipeline(temp_dir, monkeypatch) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    augmentations = [BasicAugmentation(probability=0.2)]

    def fake_build_pipeline(*_args, **_kwargs):
        return adapter, loader, loader, augmentations

    monkeypatch.setattr(pipelines_module, "build_pipeline", fake_build_pipeline)

    cfg_path = temp_dir / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "m_epochs: 1",
                "max_iterations: 1",
                "xai_enabled: false",
                "device: cpu",
                f"checkpoint_dir: {temp_dir / 'checkpoints'}",
                f"report_dir: {temp_dir / 'reports'}",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    train_result = runner.invoke(
        cli_module.app,
        [
            "train",
            "--config",
            str(cfg_path),
            "--dataset",
            "mnist",
            "--without-dashboard",
            "--max-train-samples",
            "16",
            "--max-val-samples",
            "8",
        ],
    )
    assert train_result.exit_code == 0
    assert "TRAINING COMPLETE" in train_result.output


def test_cli_train_command_without_dashboard_keeps_events_for_export(temp_dir, monkeypatch) -> None:
    loader = _gray_loader()
    model = GrayCNN()
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    augmentations = [BasicAugmentation(probability=0.2)]

    def fake_build_pipeline(*_args, **_kwargs):
        return adapter, loader, loader, augmentations

    monkeypatch.setattr(pipelines_module, "build_pipeline", fake_build_pipeline)

    cfg_path = temp_dir / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "m_epochs: 1",
                "max_iterations: 1",
                "xai_enabled: false",
                "device: cpu",
                f"checkpoint_dir: {temp_dir / 'checkpoints'}",
                f"report_dir: {temp_dir / 'reports'}",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    train_result = runner.invoke(
        cli_module.app,
        [
            "train",
            "--config",
            str(cfg_path),
            "--dataset",
            "mnist",
            "--without-dashboard",
            "--max-train-samples",
            "16",
            "--max-val-samples",
            "8",
        ],
    )
    assert train_result.exit_code == 0
    run_dirs = sorted((temp_dir / "reports").glob("run_*"))
    assert run_dirs, "Run directory should be created"
    run_dir = run_dirs[-1]
    assert (run_dir / "events.jsonl").exists()

    export_out = temp_dir / "exported_dashboard"
    export_result = runner.invoke(
        cli_module.app,
        [
            "dashboard",
            "export",
            "--run-dir",
            str(run_dir),
            "--out",
            str(export_out),
        ],
    )
    assert export_result.exit_code == 0
    assert (export_out / "index.html").exists()


def test_cli_train_command_rejects_unknown_dataset(temp_dir) -> None:
    cfg_path = temp_dir / "cfg.yaml"
    cfg_path.write_text("m_epochs: 1\nmax_iterations: 1\ndevice: cpu\nxai_enabled: false\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["train", "--config", str(cfg_path), "--dataset", "custom"])
    assert result.exit_code == 1
