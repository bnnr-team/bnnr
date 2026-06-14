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


def test_baseline_keeps_best_epoch_not_last(temp_dir, monkeypatch) -> None:
    """The baseline phase must restore its best epoch's weights, not the last."""
    import copy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import bnnr.training.loop as loop_mod
    from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter

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
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(
        m_epochs=3,
        max_iterations=0,  # baseline only
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "ckpt",
        report_dir=temp_dir / "reports",
        save_checkpoints=False,
    )

    # Script evaluate so epoch 2 (acc 0.8) is the best; epoch 3 (0.5) is last.
    scripted = [0.3, 0.8, 0.5]
    calls = {"n": 0}

    def fake_eval(trainer, ldr, cache_predictions=False):
        i = calls["n"]
        calls["n"] += 1
        acc = scripted[i] if i < len(scripted) else 0.8
        return {"accuracy": acc, "loss": 1.0 - acc, "f1_macro": acc}

    # Snapshot model weights right after each baseline train epoch.
    snapshots: list[dict] = []
    real_train = loop_mod.train_epoch

    def traced_train(trainer, ldr, augmentations=None):
        out = real_train(trainer, ldr, augmentations)
        snapshots.append(copy.deepcopy(trainer.model.state_dict()))
        return out

    monkeypatch.setattr(loop_mod, "evaluate", fake_eval)
    monkeypatch.setattr(loop_mod, "train_epoch", traced_train)

    trainer = BNNRTrainer(adapter, loader, loader, [], cfg)
    trainer.run()

    assert len(snapshots) == 3
    # trainer.model is the adapter; weights are nested under "model".
    final = trainer.model.state_dict()["model"]
    best = snapshots[1]["model"]
    last = snapshots[2]["model"]
    # Weights match the best epoch (index 1), not the last (index 2).
    assert torch.equal(final["fc.weight"], best["fc.weight"])
    assert not torch.equal(final["fc.weight"], last["fc.weight"])


def test_baseline_reeval_win_does_not_crash(temp_dir, monkeypatch) -> None:
    """A winning baseline_reeval must be treated as no-improvement, not crash."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import bnnr.training.branching as branching_mod
    from bnnr.augmentations import ChurchNoise
    from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter

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
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "ckpt",
        report_dir=temp_dir / "reports",
        save_checkpoints=False,
        reeval_baseline_per_iteration=True,
    )

    # Force the iteration to "select" the baseline re-evaluation, the exact case
    # that used to raise StopIteration at the augmentation lookup.
    def _fake_select(results, baseline_metrics, config, xai_scores=None):
        return "baseline_reeval" if "baseline_reeval" in results else None

    monkeypatch.setattr(branching_mod, "select_best_path", _fake_select)

    aug = ChurchNoise(probability=1.0, random_state=0)
    trainer = BNNRTrainer(adapter, loader, loader, [aug], cfg)
    result = trainer.run()  # must not raise

    assert result is not None
    active_names = [a.name for a in trainer._active_augmentations]
    assert "baseline_reeval" not in active_names
    assert "baseline_reeval" not in result.best_path
