"""Unit tests for training.loop orchestration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bnnr.training.loop import run_single_iteration


@pytest.fixture()
def mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.config.m_epochs = 2
    trainer.config.selection_metric = "accuracy"
    trainer.config.selection_mode = "max"
    trainer.config.candidate_pruning_warmup_epochs = 1
    trainer._active_augmentations = []
    trainer._clone_state_dict.return_value = {"w": 1}
    trainer.model.state_dict.return_value = {"w": 1}
    trainer._check_pause = MagicMock()
    trainer.train_loader = MagicMock()
    trainer.val_loader = MagicMock()
    return trainer


def test_run_single_iteration_returns_best_epoch(mock_trainer: MagicMock) -> None:
    with (
        patch("bnnr.training.loop.train_epoch") as mock_train,
        patch("bnnr.training.loop.evaluate") as mock_eval,
        patch("bnnr.training.loop._branching.should_prune_candidate", return_value=False),
    ):
        mock_train.return_value = {"loss": 0.1}
        mock_eval.side_effect = [
            {"accuracy": 0.4, "loss": 0.2},
            {"accuracy": 0.9, "loss": 0.1},
        ]
        aug = MagicMock()
        aug.name = "icd"

        metrics, state, best_epoch, pruned = run_single_iteration(
            mock_trainer,
            aug,
            baseline_metrics={"accuracy": 0.95},
        )

    assert metrics["accuracy"] == 0.9
    assert best_epoch == 2
    assert pruned is False
    assert state == {"w": 1}
    assert mock_train.call_count == 2
    assert mock_eval.call_count == 2


def test_precompute_runs_after_baseline_and_under_run_dir(temp_dir, monkeypatch) -> None:
    """XAI cache is precomputed after the baseline phase and stored under run_dir."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import bnnr.training.loop as loop_mod
    from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter
    from bnnr.icd import ICD

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(6, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.conv1(x))
            return self.fc(self.pool(x).flatten(1))

    torch.manual_seed(0)
    x = torch.rand(12, 3, 8, 8)
    y = torch.randint(0, 4, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = TinyCNN()
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
        xai_enabled=True,
        device="cpu",
        checkpoint_dir=temp_dir / "ckpt",
        report_dir=temp_dir / "reports",
        save_checkpoints=False,
    )
    icd = ICD(model=model, target_layers=[model.conv1], probability=1.0, random_state=0)

    order: list[str] = []
    real_train = loop_mod.train_epoch
    real_precompute = loop_mod._xai.precompute_xai_cache

    def traced_train(trainer, ldr, augmentations=None):
        order.append("train")
        return real_train(trainer, ldr, augmentations)

    def traced_precompute(trainer):
        order.append("precompute")
        return real_precompute(trainer)

    monkeypatch.setattr(loop_mod, "train_epoch", traced_train)
    monkeypatch.setattr(loop_mod._xai, "precompute_xai_cache", traced_precompute)

    trainer = BNNRTrainer(adapter, loader, loader, [icd], cfg)
    trainer.run()

    assert "precompute" in order
    # At least one (baseline) train epoch ran before the cache was precomputed.
    assert order.index("train") < order.index("precompute")

    run_cache = trainer.reporter.run_dir / "xai_cache"
    assert run_cache.exists()
    assert (run_cache / "manifest.json").exists()
    # Default cache must not land in the shared checkpoint_dir.
    assert not (cfg.checkpoint_dir / "xai_cache").exists()
