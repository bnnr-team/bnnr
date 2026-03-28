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
            from ultralytics.cfg import get_cfg
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
        self._model.to(self.device)
        # ``YOLO().model`` loads with all parameters ``requires_grad=False`` (inference-ready).
        for p in self._model.parameters():
            p.requires_grad = True

        # Checkpoints often store ``model.args`` as a small dict; v8 loss needs full hyperparameters
        # (``box`` / ``cls`` / ``dfl`` gains, etc.) on an attribute-style namespace.
        ckpt_args = getattr(self._model, "args", None)
        if isinstance(ckpt_args, dict) or ckpt_args is None or not hasattr(ckpt_args, "box"):
            hyp = get_cfg()
            if isinstance(ckpt_args, dict):
                for key, value in ckpt_args.items():
                    if hasattr(hyp, key):
                        setattr(hyp, key, value)
            self._model.args = hyp

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
        """Training step using Ultralytics' internal loss (v8 API).

        Ultralytics 8.x expects ``model.loss(batch_dict)`` where ``batch_dict`` has
        ``img`` (BCHW, typically 0–255 float), ``cls`` (N,1), ``bboxes`` (N,4) normalized
        xywh, and ``batch_idx`` (N,) per-target image index — not a flat (N,6) tensor.
        """
        images, targets = batch
        self._model.train()
        # ``load_state_dict`` / checkpoints must not leave Ultralytics weights frozen.
        for _p in self._model.parameters():
            _p.requires_grad_(True)

        if isinstance(images, Tensor):
            images = images.to(self.device, dtype=torch.float32)

        _b, _c, img_h, img_w = images.shape

        cls_parts: list[Tensor] = []
        bbox_parts: list[Tensor] = []
        bi_parts: list[Tensor] = []
        for i, t in enumerate(targets):
            boxes = t["boxes"]  # xyxy in pixel coords on resized image
            labels = t["labels"]
            if len(boxes) == 0:
                continue
            boxes = boxes.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device)
            x1, y1, x2, y2 = boxes.unbind(1)
            w_n = (x2 - x1) / float(img_w)
            h_n = (y2 - y1) / float(img_h)
            cx = (x1 + x2) * 0.5 / float(img_w)
            cy = (y1 + y2) * 0.5 / float(img_h)
            xywh = torch.stack((cx, cy, w_n, h_n), dim=1)
            cls_parts.append(labels.float().view(-1, 1))
            bbox_parts.append(xywh)
            bi_parts.append(
                torch.full((len(boxes),), float(i), device=self.device, dtype=torch.float32),
            )

        if cls_parts:
            cls_t = torch.cat(cls_parts, dim=0)
            bboxes_t = torch.cat(bbox_parts, dim=0)
            batch_idx_t = torch.cat(bi_parts, dim=0)
        else:
            cls_t = torch.zeros(0, 1, device=self.device, dtype=torch.float32)
            bboxes_t = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            batch_idx_t = torch.zeros(0, device=self.device, dtype=torch.float32)

        # Match Ultralytics dataloader image scale (uint8-style 0–255).
        ultra_batch = {
            "img": (images * 255.0).clamp(0.0, 255.0).contiguous(),
            "cls": cls_t,
            "bboxes": bboxes_t,
            "batch_idx": batch_idx_t,
        }

        self.optimizer.zero_grad()

        yolo_model = cast(Any, self._model)
        with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
            loss_out, _loss_items = yolo_model.loss(ultra_batch)
            total_loss = loss_out.sum() if isinstance(loss_out, Tensor) else torch.as_tensor(
                loss_out, device=self.device, dtype=torch.float32,
            )

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": float(total_loss.item())}

    def eval_step(self, batch: Any) -> dict[str, float]:
        """Evaluation step: accumulate predictions for epoch-level mAP."""
        images, targets = batch
        self._model.eval()

        if isinstance(images, Tensor):
            src = (images.float().to(self.device) * 255.0).clamp(0.0, 255.0)
        else:
            src = images
        results = self._yolo.predict(
            source=src,
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
        self._model.to(self.device)
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
