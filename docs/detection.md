# Object Detection

BNNR supports object detection as a first-class task since v0.2.0.
The same Train → Explain → Improve → Prove loop applies: train a detector, generate XAI saliency for bounding boxes, improve with detection-aware augmentations and ICD/AICD, and prove with mAP metrics and structured reports.

## Supported models

### Torchvision detectors

Any torchvision-style model whose `forward` returns losses in train mode and prediction dicts in eval mode:

- Faster R-CNN (`fasterrcnn_resnet50_fpn`)
- RetinaNet (`retinanet_resnet50_fpn`)
- SSD (`ssd300_vgg16`)
- FCOS (`fcos_resnet50_fpn`)

Wrap with `DetectionAdapter`:

```python
from torchvision.models.detection import retinanet_resnet50_fpn
from bnnr import DetectionAdapter

model = retinanet_resnet50_fpn(weights="DEFAULT")

adapter = DetectionAdapter(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    device="auto",
    score_threshold=0.05,
)
```

**Constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | required | Torchvision detection model |
| `optimizer` | required | PyTorch optimizer |
| `target_layers` | `None` | Layers for XAI saliency (e.g. `[model.backbone.body.layer4]`) |
| `device` | `"cuda"` | Device string or `"auto"` |
| `scheduler` | `None` | Optional LR scheduler |
| `use_amp` | `False` | Enable automatic mixed precision |
| `score_threshold` | `0.05` | Min confidence for predictions |

### Ultralytics YOLO

For YOLOv8 and compatible Ultralytics models, use `UltralyticsDetectionAdapter`:

```python
from bnnr.detection_adapter import UltralyticsDetectionAdapter

adapter = UltralyticsDetectionAdapter(
    model_name="yolov8n.pt",
    device="auto",
    num_classes=80,
    lr=1e-3,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"yolov8n.pt"` | Ultralytics model path or name |
| `device` | `"cuda"` | Device string |
| `score_threshold` | `0.05` | Min confidence for predictions |
| `num_classes` | `None` | Number of classes (auto-detected if `None`) |
| `lr` | `1e-3` | Learning rate (used when `optimizer` is `None`) |
| `optimizer` | `None` | Custom optimizer (built automatically if `None`) |
| `use_amp` | `False` | Enable automatic mixed precision |

## Dataset format

Detection datasets must return `(image, target, index)` tuples where:

- `image` — `torch.Tensor` of shape `(C, H, W)`, values in `[0, 1]`
- `target` — `dict` with keys:
  - `boxes` — `(N, 4)` float tensor in xyxy format
  - `labels` — `(N,)` int64 tensor (class ids, 0 = background)
  - Optional: `area`, `iscrowd`, `image_id`
- `index` — integer sample index

### Collate functions

Standard PyTorch collation doesn't work for variable-length targets. Use the provided collate functions:

```python
from bnnr import detection_collate_fn_with_index

train_loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=detection_collate_fn_with_index,
)
```

- `detection_collate_fn(batch)` → `(Tensor[B,C,H,W], list[dict])`
- `detection_collate_fn_with_index(batch)` → `(Tensor[B,C,H,W], list[dict], Tensor[B])`

## Detection augmentations

All detection augmentations are bbox-aware — they transform both the image and bounding boxes consistently.

### Built-in transforms

| Class | Description | Key parameters |
|-------|-------------|----------------|
| `DetectionHorizontalFlip` | Horizontal flip with bbox mirroring | `probability` |
| `DetectionVerticalFlip` | Vertical flip with bbox mirroring | `probability` |
| `DetectionRandomRotate90` | 90° rotation with bbox transform | `probability` |
| `DetectionRandomScale` | Random resize with bbox scaling | `probability`, `scale_range=(0.8, 1.2)` |
| `MosaicAugmentation` | 4-image mosaic composition | `probability`, `output_size=(640, 640)` |
| `DetectionMixUp` | Alpha-blend two images with merged targets | `probability`, `alpha_range=(0.3, 0.7)` |
| `AlbumentationsBboxAugmentation` | Albumentations wrapper with bbox support | `transform`, `probability` |

### Detection ICD / AICD

XAI-driven augmentations that use bounding-box saliency priors:

| Class | Behavior |
|-------|----------|
| `DetectionICD` | Masks high-saliency (object) tiles — forces the model to learn from context |
| `DetectionAICD` | Masks low-saliency (background) tiles — sharpens focus on key features |

Shared parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold_percentile` | `70.0` | Saliency threshold for tile masking |
| `tile_size` | `8` | Tile grid size in pixels |
| `fill_strategy` | `"gaussian_blur"` | Fill method: `gaussian_blur`, `local_mean`, `global_mean`, `noise`, `solid` |
| `probability` | `1.0` | Application probability |

### Presets

```python
from bnnr.detection_augmentations import get_detection_preset

augmentations = get_detection_preset("standard")  # "light", "standard", "aggressive"
```

## Configuration

Set `task="detection"` in `BNNRConfig`:

```python
from bnnr import BNNRConfig

config = BNNRConfig(
    task="detection",
    m_epochs=5,
    max_iterations=3,
    detection_bbox_format="xyxy",
    detection_score_threshold=0.5,
    metrics=["map_50", "map_50_95", "loss"],
)
```

### Detection-specific fields

| Field | Default | Description |
|-------|---------|-------------|
| `task` | `"classification"` | Set to `"detection"` |
| `detection_bbox_format` | `"xyxy"` | Box format: `xyxy`, `xywh`, or `cxcywh` |
| `detection_score_threshold` | `0.5` | Confidence threshold for evaluation |
| `detection_targets_mode` | `"auto"` | Augmentation target mode: `auto`, `image_only`, `bbox_aware` |
| `detection_class_names` | `None` | Optional list of class names for reports |

### Detection metrics

| Metric key | Description |
|------------|-------------|
| `map_50` | Mean Average Precision at IoU 0.50 |
| `map_50_95` | Mean Average Precision at IoU 0.50:0.95 |
| `loss` | Training loss |

Per-class AP and confusion matrices are computed automatically during evaluation.

## Detection XAI

BNNR generates detection-specific XAI visualizations:

- **Backbone saliency** — class-agnostic activation heatmap from the detector backbone
- **Occlusion sensitivity** — per-box saliency via systematic occlusion grids
- **Triptych panels** — ground-truth boxes, saliency overlay, and prediction boxes side by side

XAI is generated automatically when `xai_enabled=True` (default). For Ultralytics models, use `UltralyticsDetectionAdapter` to enable full XAI support; raw `ultralytics.nn.tasks` modules have limited XAI (see [troubleshooting.md §17](troubleshooting.md)).

## Full example

```python
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from bnnr import (
    BNNRConfig, BNNRTrainer, DetectionAdapter,
    DetectionHorizontalFlip, DetectionRandomScale,
    DetectionICD, detection_collate_fn_with_index,
)

model = retinanet_resnet50_fpn(weights="DEFAULT")
adapter = DetectionAdapter(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    device="auto",
)

augmentations = [
    DetectionHorizontalFlip(probability=0.5),
    DetectionRandomScale(probability=0.5, scale_range=(0.8, 1.2)),
    DetectionICD(probability=0.3),
]

config = BNNRConfig(
    task="detection",
    m_epochs=5,
    max_iterations=3,
    metrics=["map_50", "map_50_95", "loss"],
)

trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
result = trainer.run()
print(f"Best mAP@50: {result.best_metrics}")
```

## Examples and notebooks

- `examples/detection/showcase_yolo_coco128.py` — YOLO + COCO128 with auto-download, dashboard, and mAP tracking
- `examples/detection/showcase_voc.py` — Pascal VOC 2007 with full augmentation suite and XAI saliency
- `examples/detection/bnnr_detection_demo.ipynb` — interactive Jupyter notebook

## Related docs

- [configuration.md](configuration.md) — all config fields including detection
- [api_reference.md](api_reference.md) — full detection API listing
- [augmentations.md](augmentations.md) — augmentation presets and registration
- [troubleshooting.md](troubleshooting.md) — Ultralytics edge cases (§17)
