#!/usr/bin/env python3
"""Smoke test: YOLOv8 + BNNR ``generate_detection_saliency`` + JET overlay.

Run from the **repository root** with a normal dev env (``uv sync`` / ``pip install -e .[dev]``):

  python examples/detection/smoke_yolo_xai_overlay.py

Writes ``examples/detection/xai_yolov8n_activation_overlay.png`` next to this file.
Activation-based saliency is **class-agnostic** (mean conv features), so the map is
often **grainy** — unlike gradient-based CAM.  Ultralytics builds pick a conv layer
by **isotropy** (see ``generate_detection_saliency``); check printed ``col_std`` /
``row_std`` on the full-res map (both should be well above ~0 for a non-barcode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch import nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bnnr.detection_xai import (  # noqa: E402
    generate_detection_saliency,
    overlay_saliency_heatmap,
)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    img_path = out_dir / "_smoke_input_cat.jpg"
    if not img_path.is_file():
        import urllib.request

        urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
            img_path,
        )

    pil = Image.open(img_path).convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    rgb = np.asarray(pil, dtype=np.uint8).copy()
    chw = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    batch = chw.unsqueeze(0)

    yolo = YOLO("yolov8n.pt")
    torch_model = yolo.model
    if torch_model is None:
        raise RuntimeError("YOLO model is None")
    torch_model.eval()

    conv_layers = [m for m in torch_model.modules() if isinstance(m, nn.Conv2d)]
    target_layers = [conv_layers[-1]] if conv_layers else [torch_model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sal_list = generate_detection_saliency(
        torch_model,
        batch.to(device),
        target_layers,
        device=device,
        forward_layout="ultralytics_bchw",
    )
    sal = sal_list[0]
    col_std = float(np.std(sal, axis=0).mean())
    row_std = float(np.std(sal, axis=1).mean())
    print(f"saliency shape={sal.shape} col_std={col_std:.6f} row_std={row_std:.6f}")

    overlay = overlay_saliency_heatmap(rgb, sal, alpha=0.45)
    out_png = out_dir / "xai_yolov8n_activation_overlay.png"
    Image.fromarray(overlay).save(out_png)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
