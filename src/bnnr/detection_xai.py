"""Detection XAI: saliency visualizations with bounding box overlays.

Provides tools for visualizing how detection models make decisions:

- ``generate_detection_saliency``: Computes per-class saliency maps for
  detection models using the model's backbone features.
- ``draw_detection_overlay``: Draws boxes, labels, and optional saliency
  heatmap on an image.
- ``save_detection_xai_panels``: Saves three separate PNGs per sample
  (GT overlay, saliency, pred overlay) to disk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# Colour palette for drawing boxes (class_id → BGR colour)
_BOX_COLOURS = [
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def _get_box_colour(class_id: int) -> tuple[int, int, int]:
    """Return a BGR colour for the given class_id."""
    return _BOX_COLOURS[class_id % len(_BOX_COLOURS)]


# ---------------------------------------------------------------------------
#  Drawing utilities
# ---------------------------------------------------------------------------


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray | Tensor,
    labels: np.ndarray | Tensor | None = None,
    scores: np.ndarray | Tensor | None = None,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes (and optional labels/scores) on an image.

    Parameters
    ----------
    image : np.ndarray
        HWC uint8 image (BGR or RGB).
    boxes : array-like [N, 4]
        xyxy format boxes.
    labels : array-like [N] | None
        Integer class labels.
    scores : array-like [N] | None
        Confidence scores (0-1).
    class_names : list[str] | None
        Mapping from class_id to name.
    thickness : int
        Box line thickness.
    font_scale : float
        Text font scale.

    Returns
    -------
    np.ndarray – Image with boxes drawn.
    """
    out = image.copy()
    if isinstance(boxes, Tensor):
        boxes = boxes.detach().cpu().numpy()
    if labels is not None and isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()
    if scores is not None and isinstance(scores, Tensor):
        scores = scores.detach().cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        cls = int(labels[i]) if labels is not None else 0
        colour = _get_box_colour(cls)

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        # Build label text
        parts = []
        if class_names and cls < len(class_names):
            parts.append(class_names[cls])
        elif labels is not None:
            parts.append(f"cls_{cls}")
        if scores is not None:
            parts.append(f"{float(scores[i]):.2f}")
        text = " ".join(parts)

        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
            cv2.putText(
                out, text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            )

    return out


def overlay_saliency_heatmap(
    image: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a saliency heatmap on an image.

    Parameters
    ----------
    image : np.ndarray
        HWC uint8 image.
    saliency : np.ndarray
        2D float saliency map (any range; will be normalized to 0-255).
    alpha : float
        Blending factor (0 = full image, 1 = full heatmap).
    colormap : int
        OpenCV colormap constant.

    Returns
    -------
    np.ndarray – Blended image.
    """
    if saliency.ndim != 2:
        saliency = saliency.squeeze()

    # Resize saliency to match image
    h, w = image.shape[:2]
    if saliency.shape != (h, w):
        saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255
    smin, smax = saliency.min(), saliency.max()
    if smax - smin > 1e-8:
        saliency_norm = ((saliency - smin) / (smax - smin) * 255).astype(np.uint8)
    else:
        saliency_norm = np.zeros((h, w), dtype=np.uint8)

    heatmap = cv2.applyColorMap(saliency_norm, colormap)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended


# ---------------------------------------------------------------------------
#  Detection saliency generation (backbone feature-based)
# ---------------------------------------------------------------------------


def generate_detection_saliency(
    model: nn.Module,
    images: Tensor,
    target_layers: list[nn.Module],
    device: str | torch.device = "cpu",
    *,
    forward_layout: Literal["torchvision_list", "ultralytics_bchw"] = "torchvision_list",
) -> list[np.ndarray]:
    """Generate saliency maps for detection model images.

    Uses GradCAM-like approach on the backbone's target layers.
    This is a simplified version — for classification-grade XAI,
    the model needs a class-specific gradient pathway.

    For detection, we compute saliency based on the total activation
    magnitude in the target layers (class-agnostic saliency).

    Parameters
    ----------
    model : nn.Module
        Detection model.
    images : Tensor [B, C, H, W]
        Batch of images.
    target_layers : list[nn.Module]
        Layers to extract activations from.
    device : str
        Device for computation.
    forward_layout : str
        ``torchvision_list`` — ``model([CHW, ...])`` (torchvision detection).
        ``ultralytics_bchw`` — ``model(BCHW)`` with float inputs in ``[0, 1]`` (Ultralytics).

    Returns
    -------
    list[np.ndarray] – One saliency map per image, each shaped (H, W).
    """
    model.eval()
    images = images.to(device)

    activations: list[Tensor] = []

    def _hook(module: nn.Module, input: Any, output: Any) -> None:
        if isinstance(output, Tensor):
            activations.append(output.detach())

    # Register hooks
    handles = []
    for layer in target_layers:
        h = layer.register_forward_hook(_hook)
        handles.append(h)

    try:
        with torch.no_grad():
            if forward_layout == "ultralytics_bchw":
                from bnnr.detection_adapter import _det_images_to_float01

                x = _det_images_to_float01(images)
                model(x)
            else:
                # Torchvision detection: list of CHW tensors
                if images.dim() == 4:
                    image_list = [img for img in images]
                else:
                    image_list = [images]
                model(image_list)
    finally:
        for h in handles:
            h.remove()

    if not activations:
        # No activations captured — return zeros
        b = images.shape[0]
        h, w = images.shape[2], images.shape[3]
        return [np.zeros((h, w), dtype=np.float32) for _ in range(b)]

    # Use the last captured activation
    feat = activations[-1]  # [B, C, fH, fW]
    # Channel-wise mean → spatial saliency
    saliency = feat.mean(dim=1).cpu().numpy()  # [B, fH, fW]

    result = []
    for i in range(saliency.shape[0]):
        sal = saliency[i]
        # Resize to input image size
        img_h, img_w = images.shape[2], images.shape[3]
        sal_resized = cv2.resize(sal, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        result.append(sal_resized)

    return result


def _pairwise_iou(single_box_xyxy: Tensor, boxes_xyxy: Tensor) -> Tensor:
    """IoU between one box [4] and many boxes [N,4]."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    x1 = torch.maximum(single_box_xyxy[0], boxes_xyxy[:, 0])
    y1 = torch.maximum(single_box_xyxy[1], boxes_xyxy[:, 1])
    x2 = torch.minimum(single_box_xyxy[2], boxes_xyxy[:, 2])
    y2 = torch.minimum(single_box_xyxy[3], boxes_xyxy[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a = (single_box_xyxy[2] - single_box_xyxy[0]).clamp(min=0) * (single_box_xyxy[3] - single_box_xyxy[1]).clamp(min=0)
    b = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    union = a + b - inter
    return inter / (union + 1e-8)


def _matched_scores_for_queries(
    pred: dict[str, Tensor],
    query_boxes: Tensor,
    query_labels: Tensor,
    iou_threshold: float = 0.3,
) -> Tensor:
    """Return per-query matched detection scores from model predictions."""
    if query_boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    p_boxes = pred.get("boxes", torch.zeros((0, 4)))
    p_labels = pred.get("labels", torch.zeros((0,), dtype=torch.long))
    p_scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32))
    if p_boxes.numel() == 0 or p_labels.numel() == 0 or p_scores.numel() == 0:
        return torch.zeros((query_boxes.shape[0],), dtype=torch.float32)

    out = torch.zeros((query_boxes.shape[0],), dtype=torch.float32)
    for i in range(int(query_boxes.shape[0])):
        label = int(query_labels[i].item())
        mask = p_labels == label
        if not mask.any():
            out[i] = 0.0
            continue
        cand_boxes = p_boxes[mask]
        cand_scores = p_scores[mask]
        ious = _pairwise_iou(query_boxes[i], cand_boxes)
        if ious.numel() == 0:
            out[i] = 0.0
            continue
        best_idx = int(torch.argmax(ious).item())
        if float(ious[best_idx].item()) >= iou_threshold:
            out[i] = float(cand_scores[best_idx].item())
        else:
            out[i] = 0.0
    return out


def compute_detection_box_saliency_occlusion(
    model: nn.Module,
    image: Tensor,
    query_boxes: Tensor,
    query_labels: Tensor,
    *,
    baseline_pred: dict[str, Tensor] | None = None,
    predict_chw: Callable[[Tensor], dict[str, Tensor]] | None = None,
    device: str | torch.device = "cpu",
    grid_size: int = 6,
    iou_threshold: float = 0.3,
    occlusion_value: float = 0.0,
) -> tuple[list[np.ndarray], Tensor]:
    """Per-box saliency via occlusion sensitivity.

    For each queried box+class, we measure how the matched score drops when
    local regions are occluded. This is detection-conditioned and class-aware.

    predict_chw
        If set, predictions for a single ``[C,H,W]`` image come from this
        callable instead of ``model([image])[0]`` (Ultralytics / non-torchvision).
    """
    if query_boxes.numel() == 0:
        return [], torch.zeros((0,), dtype=torch.float32)

    model.eval()
    img = image.detach().to(device)
    if img.dim() != 3:
        raise ValueError("image must have shape [C,H,W]")
    _, h, w = img.shape

    qb = query_boxes.detach().to(device)
    ql = query_labels.detach().to(device)

    def _predict_one(im: Tensor) -> dict[str, Tensor]:
        if predict_chw is not None:
            return predict_chw(im)
        return model([im])[0]

    with torch.no_grad():
        base_pred = baseline_pred if baseline_pred is not None else _predict_one(img)
        baseline_scores = _matched_scores_for_queries(base_pred, qb, ql, iou_threshold=iou_threshold).to(device)

    # Coarse grid for runtime balance; upsampled to full image size.
    gh = max(2, int(grid_size))
    gw = max(2, int(grid_size))
    ph = int(np.ceil(h / gh))
    pw = int(np.ceil(w / gw))
    sal_grid = torch.zeros((int(qb.shape[0]), gh, gw), dtype=torch.float32, device=device)

    for gy in range(gh):
        y1 = gy * ph
        y2 = min(h, (gy + 1) * ph)
        if y2 <= y1:
            continue
        for gx in range(gw):
            x1 = gx * pw
            x2 = min(w, (gx + 1) * pw)
            if x2 <= x1:
                continue
            occluded = img.clone()
            occluded[:, y1:y2, x1:x2] = occlusion_value
            with torch.no_grad():
                pred_occ = _predict_one(occluded)
                occ_scores = _matched_scores_for_queries(pred_occ, qb, ql, iou_threshold=iou_threshold).to(device)
            sal_grid[:, gy, gx] = (baseline_scores - occ_scores).clamp(min=0)

    # Convert to full-resolution maps.
    maps: list[np.ndarray] = []
    for i in range(int(qb.shape[0])):
        grid_np = sal_grid[i].detach().cpu().numpy()
        up = cv2.resize(grid_np, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        maps.append(up)
    return maps, baseline_scores.detach().cpu()


# ---------------------------------------------------------------------------
#  Three-panel visualization (separate files)
# ---------------------------------------------------------------------------


def _build_detection_xai_panels(
    image: np.ndarray,
    saliency: np.ndarray | None,
    boxes_gt: np.ndarray | Tensor | None = None,
    labels_gt: np.ndarray | Tensor | None = None,
    boxes_pred: np.ndarray | Tensor | None = None,
    labels_pred: np.ndarray | Tensor | None = None,
    scores_pred: np.ndarray | Tensor | None = None,
    class_names: list[str] | None = None,
    size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build three RGB uint8 panels (each size×size), same visuals as legacy composite."""
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    panel1 = image_resized.copy()
    if boxes_gt is not None:
        if isinstance(boxes_gt, Tensor):
            boxes_gt_arr = boxes_gt.detach().cpu().numpy().copy()
        else:
            boxes_gt_arr = np.array(boxes_gt, dtype=np.float32).copy()
        boxes_gt_arr[:, [0, 2]] *= size / w
        boxes_gt_arr[:, [1, 3]] *= size / h
        panel1 = draw_boxes_on_image(
            panel1, boxes_gt_arr, labels_gt, class_names=class_names, thickness=2,
        )

    if saliency is not None:
        panel2 = overlay_saliency_heatmap(image_resized, saliency, alpha=0.5)
    else:
        panel2 = image_resized.copy()

    if saliency is not None:
        panel3 = overlay_saliency_heatmap(image_resized, saliency, alpha=0.3)
    else:
        panel3 = image_resized.copy()
    if boxes_pred is not None:
        if isinstance(boxes_pred, Tensor):
            boxes_pred_arr = boxes_pred.detach().cpu().numpy().copy()
        else:
            boxes_pred_arr = np.array(boxes_pred, dtype=np.float32).copy()
        boxes_pred_arr[:, [0, 2]] *= size / w
        boxes_pred_arr[:, [1, 3]] *= size / h
        panel3 = draw_boxes_on_image(
            panel3, boxes_pred_arr, labels_pred, scores_pred,
            class_names=class_names, thickness=2,
        )

    return panel1, panel2, panel3


def _write_panel_png(panel_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_detection_xai_panels(
    image: np.ndarray,
    saliency: np.ndarray | None,
    boxes_gt: np.ndarray | Tensor | None = None,
    labels_gt: np.ndarray | Tensor | None = None,
    boxes_pred: np.ndarray | Tensor | None = None,
    labels_pred: np.ndarray | Tensor | None = None,
    scores_pred: np.ndarray | Tensor | None = None,
    class_names: list[str] | None = None,
    save_path: Path | str = "detection_xai.png",
    size: int = 512,
) -> tuple[Path, Path, Path]:
    """Save detection XAI as three separate PNG files (GT | saliency | pred).

    *save_path* may include a suffix (e.g. ``xai_0.png``); outputs use the stem:
    ``{stem}_gt.png``, ``{stem}_sal.png``, ``{stem}_pred.png``.
    """
    stem = Path(save_path).with_suffix("")
    panel1, panel2, panel3 = _build_detection_xai_panels(
        image,
        saliency,
        boxes_gt=boxes_gt,
        labels_gt=labels_gt,
        boxes_pred=boxes_pred,
        labels_pred=labels_pred,
        scores_pred=scores_pred,
        class_names=class_names,
        size=size,
    )
    p_gt = stem.parent / f"{stem.name}_gt.png"
    p_sal = stem.parent / f"{stem.name}_sal.png"
    p_pred = stem.parent / f"{stem.name}_pred.png"
    _write_panel_png(panel1, p_gt)
    _write_panel_png(panel2, p_sal)
    _write_panel_png(panel3, p_pred)
    return p_gt, p_sal, p_pred


__all__ = [
    "draw_boxes_on_image",
    "overlay_saliency_heatmap",
    "generate_detection_saliency",
    "compute_detection_box_saliency_occlusion",
    "save_detection_xai_panels",
]
