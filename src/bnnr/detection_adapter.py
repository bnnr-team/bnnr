"""Detection model adapter for BNNR.

Provides ``DetectionAdapter`` which wraps torchvision-style detection
models (e.g. Faster R-CNN, FCOS, RetinaNet, SSD) and an optional
``UltralyticsDetectionAdapter`` for Ultralytics YOLO models.

Both adapters expose ``get_model()`` and ``get_target_layers()`` so
they satisfy the ``XAICapableModel`` protocol and can be used directly
in ``BNNRTrainer``.

Key difference from classification: detection metrics (mAP) require
epoch-level accumulation rather than per-batch computation. The adapter
stores predictions/targets during eval and computes mAP in
``epoch_end_eval()``.  The last accumulated preds/targets are kept in
``last_eval_preds`` / ``last_eval_targets`` so that the trainer can
reuse them for per-class AP and confusion matrix computation without
running a redundant evaluation pass.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

import torch
from torch import Tensor, nn

from bnnr.detection_metrics import calculate_detection_metrics
from bnnr.utils import get_device

logger = logging.getLogger(__name__)


def _targets_to_device(targets: list[dict[str, Tensor]], device: torch.device | str) -> list[dict[str, Tensor]]:
    """Move all tensors in a list of target dicts to the given device."""
    return [
        {k: v.to(device) if isinstance(v, Tensor) else v for k, v in t.items()}
        for t in targets
    ]


def _to_cpu_tensor(value: Any) -> Tensor:
    """Convert Tensor/array-like value into a CPU tensor for metric computation."""
    if isinstance(value, Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


class DetectionAdapter:
    """Adapter for torchvision-style object detection models.

    Wraps a detection model that:
    - In train mode: ``model(images, targets) -> dict[str, Tensor]`` (losses).
    - In eval mode: ``model(images) -> list[dict[str, Tensor]]`` (predictions).

    Satisfies both ``ModelAdapter`` and ``XAICapableModel`` protocols.

    Parameters
    ----------
    model : nn.Module
        Torchvision detection model (e.g. ``fasterrcnn_resnet50_fpn``).
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    target_layers : list[nn.Module] | None
        Target layers for XAI (e.g. backbone's last conv layer).
    device : str
        Device to use ('cuda', 'cpu', 'auto').
    scheduler : Any | None
        Optional LR scheduler.
    use_amp : bool
        Whether to use automatic mixed precision.
    score_threshold : float
        Minimum score for predictions during evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_layers: list[nn.Module] | None = None,
        device: str = "cuda",
        scheduler: Any | None = None,
        use_amp: bool = False,
        score_threshold: float = 0.05,
    ) -> None:
        self.device = str(get_device(device))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.target_layers = target_layers or self._auto_target_layers(model)
        self.scheduler = scheduler
        self.use_amp = use_amp and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.score_threshold = score_threshold

        # Epoch-level accumulation for mAP
        self._eval_preds: list[dict[str, Tensor]] = []
        self._eval_targets: list[dict[str, Tensor]] = []

        # Keep last completed eval for reuse by per-class metrics
        self.last_eval_preds: list[dict[str, Tensor]] = []
        self.last_eval_targets: list[dict[str, Tensor]] = []

    def _auto_target_layers(self, model: nn.Module) -> list[nn.Module]:
        """Auto-detect target layers for XAI from the model backbone."""
        # For torchvision detection models, the backbone's last conv is typical
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        return [list(model.modules())[-1]]

    def train_step(self, batch: Any) -> dict[str, float]:
        """Run one training step.

        Parameters
        ----------
        batch : tuple[Tensor, list[dict]]
            ``(images, targets)`` where images is ``Tensor[B, C, H, W]``
            and targets is a list of dicts with ``boxes`` and ``labels``.

        Returns
        -------
        dict with at least ``loss``.
        """
        self.model.train()
        images, targets = batch

        images_list = [img.to(self.device) for img in images]
        targets_on_device = _targets_to_device(targets, self.device)

        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            loss_dict = self.model(images_list, targets_on_device)
            total_loss: Tensor = sum(loss_dict.values())  # type: ignore[assignment]

        if not torch.isfinite(total_loss):
            # Rare invalid targets / unstable batches should not poison the
            # whole epoch with NaN metrics.
            logger.warning("DetectionAdapter: non-finite loss encountered; skipping batch")
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": 0.0, "loss_non_finite": 1.0}

        self.scaler.scale(total_loss).backward()

        # Guard against non-finite gradients before optimizer step.
        # This prevents corrupting model weights and cascading NaNs
        # in all subsequent batches.
        bad_grad = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            logger.warning("DetectionAdapter: non-finite gradients encountered; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": float(total_loss.item()), "loss_bad_grad": 1.0}

        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics: dict[str, float] = {"loss": float(total_loss.item())}
        for k, v in loss_dict.items():
            metrics[f"loss_{k}"] = float(v.item())
        return metrics

    def eval_step(self, batch: Any) -> dict[str, float]:
        """Run one evaluation step.

        Predictions are accumulated for epoch-level mAP computation.
        """
        self.model.eval()
        images, targets = batch

        images_list = [img.to(self.device) for img in images]

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
                predictions = self.model(images_list)

        # Filter by score threshold and move to CPU for accumulation
        for pred in predictions:
            mask = pred["scores"] >= self.score_threshold
            self._eval_preds.append({
                "boxes": pred["boxes"][mask].cpu(),
                "scores": pred["scores"][mask].cpu(),
                "labels": pred["labels"][mask].cpu(),
            })

        for target in targets:
            self._eval_targets.append({
                "boxes": target["boxes"].cpu() if isinstance(target["boxes"], Tensor) else target["boxes"],
                "labels": target["labels"].cpu() if isinstance(target["labels"], Tensor) else target["labels"],
            })

        # Return dummy loss (not available in eval mode for most detection models)
        return {"loss": 0.0}

    def epoch_end_eval(self) -> dict[str, float]:
        """Compute epoch-level detection metrics (mAP).

        Called by BNNRTrainer after all eval batches are processed.
        Saves a snapshot of accumulated preds/targets in
        ``last_eval_preds`` / ``last_eval_targets`` for reuse by
        per-class metrics, then resets the accumulation buffers.
        """
        if not self._eval_preds or not self._eval_targets:
            self.last_eval_preds = []
            self.last_eval_targets = []
            return {"map_50": 0.0, "map_50_95": 0.0}

        # Snapshot before reset so trainer can reuse for per-class AP
        self.last_eval_preds = copy.deepcopy(self._eval_preds)
        self.last_eval_targets = copy.deepcopy(self._eval_targets)

        metrics = calculate_detection_metrics(self._eval_preds, self._eval_targets)

        # Reset accumulators
        self._eval_preds = []
        self._eval_targets = []

        return metrics

    def epoch_end(self) -> None:
        """Step the LR scheduler at end of epoch."""
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


class UltralyticsDetectionAdapter:
    """Adapter for Ultralytics YOLO models.

    Wraps an Ultralytics model to work with BNNR's trainer loop.
    Satisfies both ``ModelAdapter`` and ``XAICapableModel`` protocols.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier (e.g. 'yolov8n.pt').
    device : str
        Device to use.
    score_threshold : float
        Minimum confidence for predictions.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cuda",
        score_threshold: float = 0.05,
        num_classes: int | None = None,
        lr: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
        use_amp: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for UltralyticsDetectionAdapter. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.device = str(get_device(device))
        self._yolo = YOLO(model_name)
        self._model_name = model_name
        self.score_threshold = score_threshold
        self._num_classes = num_classes

        # Epoch-level accumulation
        self._eval_preds: list[dict[str, Tensor]] = []
        self._eval_targets: list[dict[str, Tensor]] = []

        # Keep last completed eval for reuse
        self.last_eval_preds: list[dict[str, Tensor]] = []
        self.last_eval_targets: list[dict[str, Tensor]] = []

        # Internal model reference for state_dict
        model_ref = self._yolo.model
        if model_ref is None or isinstance(model_ref, str):
            raise RuntimeError("Ultralytics adapter did not expose a usable torch model.")
        self._model: nn.Module = cast(nn.Module, model_ref)

        # Optimizer — essential for weights to update during train_step
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(
                self._model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
            )

        self.use_amp = use_amp and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def train_step(self, batch: Any) -> dict[str, float]:
        """Training step using Ultralytics' internal training mechanism."""
        images, targets = batch
        self._model.train()

        if isinstance(images, Tensor):
            images = images.to(self.device)

        # Convert targets to Ultralytics format
        # Ultralytics expects: Tensor[N, 6] where columns = [batch_idx, class, cx, cy, w, h] (normalized)
        batch_targets = []
        for i, t in enumerate(targets):
            boxes = t["boxes"]  # xyxy
            labels = t["labels"]
            if len(boxes) == 0:
                continue
            # Convert xyxy to cxcywh normalized
            img_h, img_w = images.shape[-2], images.shape[-1]
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_w
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_h
            w = (boxes[:, 2] - boxes[:, 0]) / img_w
            h = (boxes[:, 3] - boxes[:, 1]) / img_h
            batch_idx = torch.full((len(boxes),), i, dtype=torch.float32)
            cls_col = labels.float()
            target_row = torch.stack([batch_idx, cls_col, cx, cy, w, h], dim=1)
            batch_targets.append(target_row)

        if batch_targets:
            batch_target_tensor = torch.cat(batch_targets, dim=0).to(self.device)
        else:
            batch_target_tensor = torch.zeros(0, 6, device=self.device)

        # Forward + loss + optimizer step
        self.optimizer.zero_grad()

        yolo_model = cast(Any, self._model)
        with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            preds = yolo_model(images)
            loss = yolo_model.loss(preds, batch_target_tensor)
            total_loss = loss if isinstance(loss, Tensor) else sum(loss.values())  # type: ignore[arg-type]

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": float(total_loss.item())}

    def eval_step(self, batch: Any) -> dict[str, float]:
        """Evaluation step: accumulate predictions for epoch-level mAP."""
        images, targets = batch
        self._model.eval()

        results = self._yolo.predict(
            source=images if isinstance(images, Tensor) else images,
            device=self.device,
            conf=self.score_threshold,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            boxes_xyxy = _to_cpu_tensor(boxes.xyxy)
            scores = _to_cpu_tensor(boxes.conf)
            labels = _to_cpu_tensor(boxes.cls).long()
            self._eval_preds.append({
                "boxes": boxes_xyxy,
                "scores": scores,
                "labels": labels,
            })

        for target in targets:
            self._eval_targets.append({
                "boxes": target["boxes"].cpu() if isinstance(target["boxes"], Tensor) else target["boxes"],
                "labels": target["labels"].cpu() if isinstance(target["labels"], Tensor) else target["labels"],
            })

        return {"loss": 0.0}

    def epoch_end_eval(self) -> dict[str, float]:
        """Compute mAP over accumulated predictions."""
        if not self._eval_preds or not self._eval_targets:
            self.last_eval_preds = []
            self.last_eval_targets = []
            return {"map_50": 0.0, "map_50_95": 0.0}

        self.last_eval_preds = copy.deepcopy(self._eval_preds)
        self.last_eval_targets = copy.deepcopy(self._eval_targets)

        metrics = calculate_detection_metrics(self._eval_preds, self._eval_targets)
        self._eval_preds = []
        self._eval_targets = []
        return metrics

    def epoch_end(self) -> None:
        """No-op: Ultralytics manages its own scheduler."""
        pass

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.use_amp and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def get_target_layers(self) -> list[nn.Module]:
        # Best-effort: find last conv layer in backbone
        conv_layers = [m for m in self._model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
        return [list(self._model.modules())[-1]]

    def get_model(self) -> nn.Module:
        return self._model


__all__ = [
    "DetectionAdapter",
    "UltralyticsDetectionAdapter",
]
