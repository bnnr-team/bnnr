"""Shared pytest fixtures and test configuration for BNNR."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.core import BNNRConfig, SimpleTorchAdapter


class DummyCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def dummy_model() -> nn.Module:
    return DummyCNN()


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    x = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 3, (10,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


@pytest.fixture
def sample_images() -> torch.Tensor:
    return torch.rand(4, 3, 32, 32)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    return torch.randint(0, 3, (4,))


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_config(tmp_path: Path) -> BNNRConfig:
    return BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        xai_enabled=False,
        device="cpu",
        save_checkpoints=True,
    )


@pytest.fixture
def model_adapter(dummy_model: nn.Module) -> SimpleTorchAdapter:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    return SimpleTorchAdapter(
        model=dummy_model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[dummy_model.conv1],
        device="cpu",
    )
