"""Example: Using BNNR with PyTorch Lightning / HuggingFace Accelerate.

BNNR's ``ModelAdapter`` protocol is intentionally minimal so that any
training framework can provide an adapter.  This file shows two
approaches:

1. **LightningAdapter** – wraps an existing ``pl.LightningModule`` so
   that BNNR can drive the augmentation search while Lightning handles
   device placement, mixed precision, etc.

2. **AccelerateAdapter** – wraps an Accelerate-prepared model/optimizer
   pair for the same purpose.

Neither Lightning nor Accelerate are hard dependencies of BNNR.
Install them only if you need this integration::

    pip install pytorch-lightning accelerate
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from bnnr.utils import calculate_metrics

# ---------------------------------------------------------------------------
# 1.  Lightning Adapter
# ---------------------------------------------------------------------------

class LightningAdapter:
    """Adapter that delegates train/eval steps to a LightningModule.

    Usage::

        import pytorch_lightning as pl

        class MyModule(pl.LightningModule):
            ...

        lit_model = MyModule()
        adapter = LightningAdapter(lit_model, target_layers=[lit_model.backbone[-1]])

        trainer = BNNRTrainer(adapter, train_loader, val_loader, augs, config)
        result = trainer.run()
    """

    def __init__(
        self,
        lightning_module: Any,  # pl.LightningModule (not typed to avoid hard dep)
        target_layers: list[nn.Module] | None = None,
        eval_metrics: list[str] | None = None,
    ) -> None:
        self.module = lightning_module
        self.target_layers = target_layers or []
        self.eval_metrics = eval_metrics or ["accuracy", "f1_macro"]
        # Ensure the module's optimizer is available
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            result = self.module.configure_optimizers()
            if isinstance(result, torch.optim.Optimizer):
                self._optimizer = result
            elif isinstance(result, (list, tuple)):
                if isinstance(result[0], torch.optim.Optimizer):
                    self._optimizer = result[0]
                elif isinstance(result[0], (list, tuple)):
                    self._optimizer = result[0][0]
                if len(result) > 1 and result[1]:
                    schedulers = result[1]
                    if isinstance(schedulers, (list, tuple)):
                        self._scheduler = schedulers[0]
                    else:
                        self._scheduler = schedulers
            elif isinstance(result, dict):
                self._optimizer = result["optimizer"]
                self._scheduler = result.get("lr_scheduler")
            if self._optimizer is None:
                raise RuntimeError("Could not extract optimizer from configure_optimizers()")
        return self._optimizer

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.module.train()
        images, labels = batch
        device = next(self.module.parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        optimizer = self._ensure_optimizer()
        optimizer.zero_grad()

        logits = self.module(images)
        loss = self.module.training_step((images, labels), 0) if hasattr(self.module, "training_step") else nn.functional.cross_entropy(logits, labels)
        if isinstance(loss, dict):
            loss_val = loss["loss"]
        elif isinstance(loss, Tensor):
            loss_val = loss
        else:
            loss_val = loss

        loss_val.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        metrics = calculate_metrics(preds, labels.detach().cpu().numpy(), metrics=self.eval_metrics)
        metrics["loss"] = float(loss_val.item())
        return metrics

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.module.eval()
        images, labels = batch
        device = next(self.module.parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = self.module(images)
            loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        metrics = calculate_metrics(preds, labels.detach().cpu().numpy(), metrics=self.eval_metrics)
        metrics["loss"] = float(loss.item())
        return metrics

    def epoch_end(self) -> None:
        if self._scheduler is not None:
            self._scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"model": self.module.state_dict()}
        if self._optimizer is not None:
            state["optimizer"] = self._optimizer.state_dict()
        if self._scheduler is not None:
            state["scheduler"] = self._scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state["model"])
        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
        if self._scheduler is not None and "scheduler" in state:
            self._scheduler.load_state_dict(state["scheduler"])

    def get_target_layers(self) -> list[nn.Module]:
        return self.target_layers

    def get_model(self) -> nn.Module:
        return self.module


# ---------------------------------------------------------------------------
# 2.  Accelerate Adapter
# ---------------------------------------------------------------------------

class AccelerateAdapter:
    """Adapter for HuggingFace Accelerate-prepared models.

    Usage::

        from accelerate import Accelerator

        accelerator = Accelerator()
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        adapter = AccelerateAdapter(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            accelerator=accelerator,
            target_layers=[model.features[-1]],
        )

        trainer = BNNRTrainer(adapter, train_loader, val_loader, augs, config)
        result = trainer.run()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        accelerator: Any,  # accelerate.Accelerator (not typed to avoid hard dep)
        target_layers: list[nn.Module] | None = None,
        eval_metrics: list[str] | None = None,
        scheduler: Any = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accelerator = accelerator
        self.target_layers = target_layers or []
        self.eval_metrics = eval_metrics or ["accuracy", "f1_macro"]
        self.scheduler = scheduler

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.model.train()
        images, labels = batch
        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            logits = self.model(images)
            loss = self.criterion(logits, labels)
        self.accelerator.backward(loss)
        self.optimizer.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        metrics = calculate_metrics(preds, labels.detach().cpu().numpy(), metrics=self.eval_metrics)
        metrics["loss"] = float(loss.item())
        return metrics

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        self.model.eval()
        images, labels = batch
        with torch.no_grad():
            with self.accelerator.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        metrics = calculate_metrics(preds, labels.detach().cpu().numpy(), metrics=self.eval_metrics)
        metrics["loss"] = float(loss.item())
        return metrics

    def epoch_end(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # Accelerate wraps models; unwrap for clean state
        unwrapped = self.accelerator.unwrap_model(self.model)
        state: dict[str, Any] = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

    def get_target_layers(self) -> list[nn.Module]:
        return self.target_layers

    def get_model(self) -> nn.Module:
        return self.accelerator.unwrap_model(self.model)
