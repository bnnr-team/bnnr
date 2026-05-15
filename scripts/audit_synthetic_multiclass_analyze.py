#!/usr/bin/env python3
"""Audit helper: synthetic K-class classification + analyze_model vs manual accuracy.

Run from repo root:
  PYTHONPATH=src ./.venv/bin/python scripts/audit_synthetic_multiclass_analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from bnnr.adapter import SimpleTorchAdapter
from bnnr.analyze import analyze_model
from bnnr.core import BNNRConfig
from bnnr.evaluation import run_evaluation


def _indexed_loader(*, n: int, n_classes: int, seed: int, batch_size: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 32, 32, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)

    class IndexedDataset(TensorDataset):
        def __getitem__(self, index: int):
            a, b = super().__getitem__(index)
            return a, b, index

    ds = IndexedDataset(x, y)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        indices = torch.tensor([b[2] for b in batch])
        return imgs, labels, indices

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False)


def main() -> int:
    n_classes = 5
    n = 200
    batch_size = 16
    seed = 12345

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv1(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = TinyCNN()
    torch.manual_seed(seed)
    loader = _indexed_loader(n=n, n_classes=n_classes, seed=seed, batch_size=batch_size)

    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )
    cfg = BNNRConfig(device="cpu", task="classification", m_epochs=1, max_iterations=1)

    metrics_ref, _, _, _, _ = run_evaluation(
        adapter, loader, cfg, return_preds_labels=False
    )
    manual_acc = float(metrics_ref["accuracy"])

    report = analyze_model(
        adapter,
        loader,
        config=cfg,
        output_dir=None,
        run_data_quality=False,
        xai_enabled=False,
        cv_folds=0,
    )
    reported = float(report.metrics.get("accuracy", -1.0))
    delta = abs(reported - manual_acc)

    print("Synthetic multi-class audit (K=%d, n=%d)" % (n_classes, n))
    print("  reference accuracy (run_evaluation): %.6f" % manual_acc)
    print("  analyze_model metrics['accuracy']:    %.6f" % reported)
    print("  |delta|: %.6f" % delta)
    if delta > 1e-5:
        print("  WARNING: accuracy mismatch above tolerance")
        return 1
    print("  OK: accuracies match (same code path as analyze_model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
