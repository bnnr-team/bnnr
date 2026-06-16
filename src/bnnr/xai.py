"""XAI explainers and saliency generation utilities used by BNNR."""

from __future__ import annotations

import abc
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from bnnr.craft import (
    CRAFTExplainer,
    NMFConceptExplainer,
    RealCRAFTExplainer,
    RecursiveCRAFTExplainer,
)
from bnnr.utils import lazy_cv2 as cv2


class BaseExplainer(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def explain(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> np.ndarray:
        raise NotImplementedError

    def visualize(self, images: np.ndarray, maps: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlays: list[np.ndarray] = []
        for image, sal in zip(images, maps):
            if image.ndim == 2:
                image = image[..., None]
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=2)

            if sal.shape[:2] != image.shape[:2]:
                sal = cv2.resize(sal.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            heatmap = cv2.applyColorMap((sal_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            merged = np.clip((1 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(np.float32), 0, 255)
            overlays.append(merged.astype(np.uint8))
        return np.stack(overlays, axis=0)

    def save_visualizations(self, images: np.ndarray, maps: np.ndarray, save_dir: Path, prefix: str = "xai") -> list[Path]:
        save_dir.mkdir(parents=True, exist_ok=True)
        overlays = self.visualize(images, maps)
        paths: list[Path] = []
        for idx, overlay in enumerate(overlays):
            p = save_dir / f"{prefix}_{idx}.png"
            cv2.imwrite(str(p), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths


class GradCAMExplainer(BaseExplainer):
    """Grad-CAM (eigen-smoothed) saliency via the pytorch-grad-cam backend."""

    name = "gradcam"

    def __init__(
        self,
        use_cuda: bool = True,
        batch_size: int = 16,
        eigen_smooth: bool = True,
        aug_smooth: bool = False,
    ) -> None:
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth

    def explain(self, model: nn.Module, images: Tensor, labels: Tensor, target_layers: list[nn.Module]) -> np.ndarray:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError as exc:
            raise RuntimeError("pytorch-grad-cam is required for GradCAMExplainer") from exc

        device = images.device
        model = model.to(device)
        targets = [
            ClassifierOutputTarget(int(label.item()) if label.ndim == 0 else int(label.argmax().item()))
            for label in labels
        ]

        cam_kwargs: dict[str, Any] = {
            "model": model,
            "target_layers": target_layers,
        }
        # `grad-cam` changed API and removed `use_cuda` in newer releases.
        if "use_cuda" in inspect.signature(GradCAM.__init__).parameters:
            cam_kwargs["use_cuda"] = self.use_cuda and torch.cuda.is_available()

        with GradCAM(**cam_kwargs) as cam:
            grayscale_cam = cam(
                input_tensor=images,
                targets=targets,
                eigen_smooth=self.eigen_smooth,
                aug_smooth=self.aug_smooth,
            )
        out = np.asarray(grayscale_cam, dtype=np.float32)
        out = np.clip(out, 0.0, 1.0)
        return out


class OptiCAMExplainer(BaseExplainer):
    """Opti-CAM saliency (Zhang et al., 2023).

    Instead of weighting activation maps by gradients, Opti-CAM optimises a
    per-image combination of the target-layer activations so that masking the
    input with the resulting map maximises the model's confidence in the target
    class. The activations are captured once and held fixed; only the
    combination weights are optimised, so no model gradients are needed.
    """

    name = "opticam"

    def __init__(
        self,
        use_cuda: bool = True,
        batch_size: int = 16,
        n_iters: int = 100,
        lr: float = 0.1,
    ) -> None:
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.lr = lr

    @staticmethod
    def _combine(activations: Tensor, weights: Tensor, out_hw: tuple[int, int]) -> Tensor:
        # Convex combination of activation maps, upsampled and min-max normalised
        # to [0, 1] so it can be used as a soft multiplicative mask.
        alpha = torch.softmax(weights, dim=1)
        combined = (alpha.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
        upsampled = torch.nn.functional.interpolate(
            combined.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False
        ).squeeze(1)
        flat_min = upsampled.amin(dim=(1, 2), keepdim=True)
        flat_max = upsampled.amax(dim=(1, 2), keepdim=True)
        return (upsampled - flat_min) / (flat_max - flat_min + 1e-8)

    def explain(self, model: nn.Module, images: Tensor, labels: Tensor, target_layers: list[nn.Module]) -> np.ndarray:
        if not target_layers:
            raise ValueError("OptiCAMExplainer requires at least one target layer")

        device = images.device
        if self.use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        model = model.to(device)
        images = images.to(device)
        batch, _, height, width = images.shape

        targets = torch.as_tensor(
            [int(label.item()) if label.ndim == 0 else int(label.argmax().item()) for label in labels],
            device=device,
            dtype=torch.long,
        )

        captured: list[Tensor] = []
        handles = [layer.register_forward_hook(lambda _m, _i, out: captured.append(out)) for layer in target_layers]

        was_training = model.training
        model.eval()
        prev_requires_grad = [p.requires_grad for p in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(False)
        try:
            with torch.no_grad():
                model(images)
            for handle in handles:
                handle.remove()
            handles = []

            feat_h = max(act.shape[2] for act in captured)
            feat_w = max(act.shape[3] for act in captured)
            activations = torch.cat(
                [
                    act
                    if act.shape[2:] == (feat_h, feat_w)
                    else torch.nn.functional.interpolate(act, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
                    for act in captured
                ],
                dim=1,
            ).detach()

            weights = torch.zeros(batch, activations.shape[1], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([weights], lr=self.lr)
            for _ in range(self.n_iters):
                optimizer.zero_grad()
                mask = self._combine(activations, weights, (height, width))
                masked = mask.unsqueeze(1) * images
                probs = torch.softmax(model(masked), dim=1)
                score = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                (-score.mean()).backward()
                optimizer.step()

            with torch.no_grad():
                saliency = self._combine(activations, weights, (height, width))
            return saliency.detach().cpu().numpy().astype(np.float32)
        finally:
            for handle in handles:
                handle.remove()
            for param, requires in zip(model.parameters(), prev_requires_grad):
                param.requires_grad_(requires)
            if was_training:
                model.train()


def generate_saliency_maps(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_layers: list[nn.Module],
    method: str = "opticam",
    **kwargs: Any,
) -> np.ndarray:
    method = method.lower()
    if method == "opticam":
        return OptiCAMExplainer(**kwargs).explain(model, images, labels, target_layers)
    if method == "gradcam":
        return GradCAMExplainer(**kwargs).explain(model, images, labels, target_layers)
    if method == "real_craft":
        return RealCRAFTExplainer(**kwargs).explain(model, images, labels, target_layers)
    if method == "craft":
        return CRAFTExplainer(**kwargs).explain(model, images, labels, target_layers)
    if method == "nmf_concepts" or method == "nmf":
        return NMFConceptExplainer(**kwargs).explain(model, images, labels, target_layers)
    raise ValueError(f"Unknown XAI method: {method}")


def generate_craft_concepts(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_layers: list[nn.Module],
    n_concepts: int = 10,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[int, float]]:
    explainer = CRAFTExplainer(n_concepts=n_concepts, **kwargs)
    maps = explainer.explain(model, images, labels, target_layers)
    scores = explainer.get_concept_importance(model, images, labels, target_layers)
    return maps, scores


def generate_nmf_concepts(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_layers: list[nn.Module],
    n_concepts: int = 10,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[int, float]]:
    explainer = NMFConceptExplainer(n_concepts=n_concepts, **kwargs)
    maps = explainer.explain(model, images, labels, target_layers)
    scores = explainer.get_concept_importance(model, images, labels, target_layers)
    return maps, scores


def save_xai_visualization(
    images: np.ndarray,
    maps: np.ndarray,
    save_dir: Path,
    prefix: str = "xai",
    format: str = "png",
    output_size: int | None = None,
) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    explainer = OptiCAMExplainer()
    overlays = explainer.visualize(images, maps)
    paths: list[Path] = []
    for idx, overlay in enumerate(overlays):
        if output_size is not None and output_size > 0:
            overlay = cv2.resize(overlay, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        path = save_dir / f"{prefix}_{idx}.{format}"
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


__all__ = [
    "BaseExplainer",
    "GradCAMExplainer",
    "OptiCAMExplainer",
    "NMFConceptExplainer",
    "CRAFTExplainer",
    "RealCRAFTExplainer",
    "RecursiveCRAFTExplainer",
    "generate_saliency_maps",
    "generate_craft_concepts",
    "generate_nmf_concepts",
    "save_xai_visualization",
]
