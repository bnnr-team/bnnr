"""Model adapter protocols and reference PyTorch adapter implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from bnnr.utils import calculate_metrics, get_device


@runtime_checkable
class ModelAdapter(Protocol):
    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        ...

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


@runtime_checkable
class XAICapableModel(ModelAdapter, Protocol):
    def get_target_layers(self) -> list[nn.Module]:
        ...

    def get_model(self) -> nn.Module:
        ...


class SimpleTorchAdapter:
    """Standard adapter wrapping a PyTorch model, criterion, optimizer.

    Supports optional LR scheduler, AMP (Automatic Mixed Precision),
    and multi-label classification (``multilabel=True``).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_layers: list[nn.Module] | None = None,
        device: str = "cuda",
        eval_metrics: list[str] | None = None,
        scheduler: Any | None = None,
        use_amp: bool = False,
        multilabel: bool = False,
        multilabel_threshold: float = 0.5,
    ) -> None:
        self.device = str(get_device(device))
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_layers = target_layers or self._auto_target_layers(model)
        self.multilabel = multilabel
        self.multilabel_threshold = multilabel_threshold
        if eval_metrics is not None:
            self.eval_metrics = eval_metrics
        elif multilabel:
            self.eval_metrics = ["f1_samples", "f1_macro", "accuracy"]
        else:
            self.eval_metrics = ["accuracy", "f1_macro"]
        self.scheduler = scheduler
        self.use_amp = use_amp and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _auto_target_layers(self, model: nn.Module) -> list[nn.Module]:
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        return [list(model.modules())[-1]]

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.model.train()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        # Squeeze extra dims (e.g. MedMNIST returns [batch, 1])
        if not self.multilabel and labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        self.optimizer.zero_grad()

        if self.multilabel:
            labels = labels.float()

        with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.multilabel:
            preds = (torch.sigmoid(logits) >= self.multilabel_threshold).int().detach().cpu().numpy()
            targets_np = labels.int().detach().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            targets_np = labels.detach().cpu().numpy()
        metrics = calculate_metrics(preds, targets_np, metrics=self.eval_metrics)
        metrics["loss"] = float(loss.item())
        return metrics

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.model.eval()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        # Squeeze extra dims (e.g. MedMNIST returns [batch, 1])
        if not self.multilabel and labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if self.multilabel:
            labels = labels.float()
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            if self.multilabel:
                preds = (torch.sigmoid(logits) >= self.multilabel_threshold).int().detach().cpu().numpy()
                targets_np = labels.int().detach().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                targets_np = labels.detach().cpu().numpy()
            metrics = calculate_metrics(preds, targets_np, metrics=self.eval_metrics)
            metrics["loss"] = float(loss.item())
        return metrics

    def epoch_end(self) -> None:
        """Call at the end of each epoch to step the LR scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.use_amp and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def get_target_layers(self) -> list[nn.Module]:
        return self.target_layers

    def get_model(self) -> nn.Module:
        return self.model


__all__ = [
    "ModelAdapter",
    "XAICapableModel",
    "SimpleTorchAdapter",
]
