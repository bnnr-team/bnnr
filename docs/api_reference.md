# Python API Reference

## What you will find here
User-facing Python API for integrating BNNR with your own model and dataloaders.

## When to use this page
Use this when CLI presets are not enough and you need full control.

## Source of truth

This page documents only symbols exported publicly from `src/bnnr/__init__.py`.

## Core training API

- `BNNRConfig`
- `BNNRTrainer`
- `quick_run`
- `BNNRRunResult`
- `CheckpointInfo`

## Model adapter API

- `ModelAdapter`
- `XAICapableModel`
- `SimpleTorchAdapter`
- `DetectionAdapter`

## Analysis API (standalone model diagnostics)

- `analyze_model` — run full analysis (metrics, XAI, data quality, failure analysis, patterns, recommendations) on an adapter and validation loader; returns `AnalysisReport`.
- `AnalysisReport` — dataclass with attributes: `metrics`, `per_class_accuracy`, `confusion`, `xai_insights`, `xai_diagnoses`, `xai_quality_summary`, `data_quality_result`, `failure_patterns`, `recommendations`, `executive_summary`, `findings`, `recommendations_structured`, `class_diagnostics`; methods: `save(output_dir)`, `to_html(path)`, `failure_patterns_list()`.

See `analyze.md` for usage and CLI.

## Reporting and events API

- `Reporter`
- `load_report`
- `compare_runs`
- `JsonlEventSink`
- `EVENT_SCHEMA_VERSION`
- `replay_events`

## Config helpers

- `load_config`
- `save_config`
- `validate_config`
- `merge_configs`
- `apply_xai_preset`
- `get_xai_preset`
- `list_xai_presets`

## Augmentation API

- `BaseAugmentation`
- `AugmentationRegistry`
- `AugmentationRunner`
- `TorchvisionAugmentation`
- `KorniaAugmentation`
- `AlbumentationsAugmentation`
- `create_kornia_pipeline`
- `kornia_available`
- `albumentations_available`

Built-in classification augmentations:

- `ChurchNoise`
- `BasicAugmentation`
- `DifPresets`
- `Drust`
- `LuxferGlass`
- `ProCAM`
- `Smugs`
- `TeaStains`

Preset helpers:

- `auto_select_augmentations`
- `get_preset`
- `list_presets`

## XAI API (classification)

Explainers and generation:

- `BaseExplainer`
- `OptiCAMExplainer`
- `NMFConceptExplainer`
- `CRAFTExplainer`
- `RealCRAFTExplainer`
- `RecursiveCRAFTExplainer`
- `generate_saliency_maps`
- `generate_craft_concepts`
- `generate_nmf_concepts`
- `save_xai_visualization`

Analysis and scoring:

- `analyze_xai_batch`
- `analyze_xai_batch_rich`
- `compute_xai_quality_score`
- `generate_class_diagnosis`
- `generate_class_insight`
- `generate_epoch_summary`
- `generate_rich_epoch_summary`

Cache:

- `XAICache`

ICD variants:

- `ICD`
- `AICD`

## Detection API

Augmentations and presets:

- `BboxAwareAugmentation`
- `AlbumentationsBboxAugmentation`
- `DetectionHorizontalFlip`
- `DetectionVerticalFlip`
- `DetectionRandomRotate90`
- `DetectionRandomScale`
- `MosaicAugmentation`
- `DetectionMixUp`
- `get_detection_preset`

XAI-driven detection augmentations:

- `DetectionICD`
- `DetectionAICD`

Data helpers and metrics:

- `detection_collate_fn`
- `detection_collate_fn_with_index`
- `calculate_detection_metrics`

## Dashboard helper

- `start_dashboard`

## Minimal classification integration

```python
import torch
import torch.nn as nn
from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, auto_select_augmentations

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

config = BNNRConfig(m_epochs=3, max_iterations=2, device="auto")
trainer = BNNRTrainer(adapter, train_loader, val_loader, auto_select_augmentations(), config)
result = trainer.run()
print(result.best_metrics)
```

## Minimal detection integration

```python
import torch
from bnnr import BNNRConfig, BNNRTrainer
from bnnr import DetectionAdapter, DetectionHorizontalFlip

model = ...
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
adapter = DetectionAdapter(model=model, optimizer=optimizer, device="cpu")

config = BNNRConfig(task="detection", m_epochs=1, max_iterations=1, device="cpu")
augmentations = [DetectionHorizontalFlip(probability=0.5, random_state=42)]

trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
result = trainer.run()
```

Detection dataloader contract:

- batch item: `(image, target, index)`
- `target["boxes"]`: `FloatTensor[N,4]`
- `target["labels"]`: `IntTensor[N]`

## `quick_run()` helper

`quick_run()` builds `SimpleTorchAdapter` internally.

```python
from bnnr import quick_run

result = quick_run(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
)
```

Useful arguments include `augmentations`, `config`/overrides, `criterion`, `optimizer`, `target_layers`, and `eval_metrics`.
