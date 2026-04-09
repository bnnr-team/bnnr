# Golden Path: Integrate BNNR Into Your Project

## What you will find here
Production-style, code-first templates for the three supported tasks:
- classification,
- multi-label,
- object detection.

Each template includes dashboard-first training flow.

## When to use this page
Use this after quickstart, when you integrate BNNR into your own training stack.

## Shared prerequisites

- Dataloader outputs include sample index (`index`) for XAI/event tracking.
- Keep `report_dir` and `checkpoint_dir` explicit per run.
- Start dashboard before `trainer.run()` if you want live monitoring.

## 1) Classification integration (dashboard-first)

### Checklist

1. Build dataloaders returning `(image, label, index)`.
2. Wrap model with `SimpleTorchAdapter` and provide `target_layers`.
3. Use a preset from `get_preset(...)`.
4. Start dashboard, then run training.

### Template

```python
import torch
import torch.nn as nn
from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, get_preset, start_dashboard

model = ...
train_loader = ...  # (image, label, index)
val_loader = ...

adapter = SimpleTorchAdapter(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    target_layers=[...],
    device="auto",
)

config = BNNRConfig(
    task="classification",
    m_epochs=5,
    max_iterations=4,
    selection_metric="accuracy",
    selection_mode="max",
    report_dir="reports/classification_run",
    checkpoint_dir="checkpoints/classification_run",
)

augmentations = get_preset("standard", random_state=config.seed)

dashboard_url = start_dashboard(config.report_dir, port=8080, auto_open=True)
print("Dashboard:", dashboard_url)

result = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config).run()
print(result.best_metrics)
```

## 2) Multi-label integration (dashboard-first)

### Checklist

1. Build dataloaders returning `(image, multi_hot_label, index)`.
2. Use `task="multilabel"` and `SimpleTorchAdapter(multilabel=True)`.
3. Use `BCEWithLogitsLoss` and raw logits model head.
4. Validate `f1_samples` and related multi-label metrics.

### Template

```python
import torch
import torch.nn as nn
from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, get_preset, start_dashboard

model = ...
train_loader = ...  # (image, multi_hot_label, index)
val_loader = ...

adapter = SimpleTorchAdapter(
    model=model,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    target_layers=[...],
    device="auto",
    multilabel=True,
    multilabel_threshold=0.5,
)

config = BNNRConfig(
    task="multilabel",
    m_epochs=3,
    max_iterations=2,
    report_dir="reports/multilabel_run",
    checkpoint_dir="checkpoints/multilabel_run",
)

augmentations = get_preset("standard", random_state=config.seed)

dashboard_url = start_dashboard(config.report_dir, port=8080, auto_open=True)
print("Dashboard:", dashboard_url)

result = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config).run()
print(result.best_metrics)  # f1_samples, f1_macro, accuracy, loss
```

## 3) Detection integration (dashboard-first)

### Checklist

1. Build dataloaders returning `(image, target_dict, index)` with `detection_collate_fn_with_index`.
2. Wrap model with `DetectionAdapter` (torchvision) or `UltralyticsDetectionAdapter` (YOLO).
3. Use bbox-aware augmentations from `get_detection_preset(...)` or build a custom list.
4. Set `task="detection"` in config. Start dashboard, then run training.

### Template

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from bnnr import (
    BNNRConfig, BNNRTrainer, DetectionAdapter,
    DetectionHorizontalFlip, DetectionRandomScale, DetectionICD,
    detection_collate_fn_with_index, start_dashboard,
)

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
train_loader = ...  # (image, {"boxes": ..., "labels": ...}, index)
val_loader = ...    # use collate_fn=detection_collate_fn_with_index

adapter = DetectionAdapter(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    device="auto",
)

config = BNNRConfig(
    task="detection",
    m_epochs=5,
    max_iterations=3,
    report_dir="reports/detection_run",
    checkpoint_dir="checkpoints/detection_run",
)

augmentations = [
    DetectionHorizontalFlip(probability=0.5),
    DetectionRandomScale(probability=0.5, scale_range=(0.8, 1.2)),
    DetectionICD(probability=0.3),
]

dashboard_url = start_dashboard(config.report_dir, port=8080, auto_open=True)
print("Dashboard:", dashboard_url)

result = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config).run()
print(result.best_metrics)  # map_50, map_50_95, loss
```

See [detection.md](detection.md) for the full guide (adapters, dataset format, augmentations, metrics, XAI).

## 4) Dashboard operational notes

- `start_dashboard(...)` prints Local URL, Network URL, and terminal QR code.
- For mobile, phone must be on same network.
- For secured controls, use token via CLI (`--dashboard-token`) or environment (`BNNR_DASHBOARD_TOKEN`) in serve mode.
- Export for offline sharing:

```bash
python3 -m bnnr dashboard export --run-dir <run_dir> --out exported_dashboard
```

## 5) Hardening checklist

- Fix `seed` for reproducibility.
- Version-control every YAML/config used in runs.
- Archive `report.json`, `events.jsonl`, and exported dashboard for each milestone.
- Add smoke checks for your integration path in CI.
