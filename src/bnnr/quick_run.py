"""High-level convenience API to execute a minimal BNNR run."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentations import BaseAugmentation
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.reporting import BNNRRunResult, Reporter


def quick_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    augmentations: list[BaseAugmentation] | None = None,
    config: BNNRConfig | None = None,
    criterion: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    target_layers: list[nn.Module] | None = None,
    eval_metrics: list[str] | None = None,
    dashboard: bool | None = None,
    **overrides: object,
) -> BNNRRunResult:
    cfg = config or BNNRConfig()
    if overrides:
        cfg = BNNRConfig(**{**cfg.model_dump(), **overrides})

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=target_layers,
        device=cfg.device,
        eval_metrics=eval_metrics or [m for m in cfg.metrics if m != "loss"],
    )

    if augmentations is None:
        from bnnr.presets import auto_select_augmentations

        augmentations = auto_select_augmentations(random_state=cfg.seed)

    _ = dashboard  # Kept for API compatibility in v0.1.
    reporter = Reporter(cfg.report_dir)

    trainer = BNNRTrainer(
        model=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=augmentations,
        config=cfg,
        reporter=reporter,
    )
    return trainer.run()
