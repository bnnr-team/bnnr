"""ConsoleReporter gates human-readable output on a single verbose flag."""

from __future__ import annotations

from bnnr.console import ConsoleReporter


def test_verbose_reporter_prints(capsys) -> None:
    ConsoleReporter(True).print("hello")
    assert capsys.readouterr().out == "hello\n"


def test_quiet_reporter_suppresses(capsys) -> None:
    ConsoleReporter(False).print("hidden")
    assert capsys.readouterr().out == ""


def test_trainer_console_follows_config_verbose() -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter

    x = torch.rand(4, 3, 8, 8)
    y = torch.randint(0, 2, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
    )
    quiet = BNNRTrainer(adapter, loader, loader, [], BNNRConfig(device="cpu", verbose=False))
    assert quiet.console.verbose is False
    loud = BNNRTrainer(adapter, loader, loader, [], BNNRConfig(device="cpu", verbose=True))
    assert loud.console.verbose is True
