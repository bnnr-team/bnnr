# Configuration (`BNNRConfig`)

## What you will find here
All configuration fields currently implemented in `src/bnnr/core.py`, grouped by responsibility, with defaults and validation notes.

## When to use this page
Use this when creating or reviewing YAML config files for CLI or Python API runs.

`BNNRConfig` is immutable (`frozen=True`), so runtime overrides use copies (`model_copy(update=...)`).

## Minimal config

```yaml
m_epochs: 1
max_iterations: 1
metrics: [accuracy, f1_macro, loss]
selection_metric: accuracy
selection_mode: max
checkpoint_dir: checkpoints
report_dir: reports
xai_enabled: false
device: auto
seed: 42
```

## Core training fields

- `m_epochs` (default: `5`)
- `max_iterations` (default: `10`)
- `metrics` (default: `['accuracy', 'f1_macro', 'loss']`)
- `selection_metric` (default: `accuracy`) — the metric used to select the best augmentation branch. Can be any metric from the tables below.
- `selection_mode` (`max` or `min`, default: `max`) — use `min` for metrics where lower is better (e.g. `loss`, `zero_one_loss`).
- `early_stopping_patience` (default: `2`)
- `device` (`cuda`, `cpu`, `auto`; default: `cuda`)
- `seed` (default: `42`)
- `save_checkpoints` (default: `true`)
- `verbose` (default: `true`)
- `log_file` (default: `null`)

## Available metrics

Any metric listed below can be used both in the `metrics` list **and** as `selection_metric`.

### Classification (single-label) metrics

| Metric name | Description | Mode |
|---|---|---|
| `accuracy` | Overall accuracy | `max` |
| `balanced_accuracy` | Mean per-class recall (handles imbalance) | `max` |
| `f1_macro` | F1 score, macro-averaged | `max` |
| `f1_micro` | F1 score, micro-averaged | `max` |
| `f1_weighted` | F1 score, weighted by support | `max` |
| `fbeta_<β>` | Fβ score (e.g. `fbeta_0.5`, `fbeta_2`), macro-averaged | `max` |
| `precision` / `precision_macro` | Precision, macro-averaged | `max` |
| `precision_micro` | Precision, micro-averaged | `max` |
| `precision_weighted` | Precision, weighted by support | `max` |
| `recall` / `recall_macro` | Recall, macro-averaged | `max` |
| `recall_micro` | Recall, micro-averaged | `max` |
| `recall_weighted` | Recall, weighted by support | `max` |
| `cohen_kappa` | Cohen's kappa coefficient | `max` |
| `mcc` | Matthews correlation coefficient | `max` |
| `jaccard_macro` | Jaccard index, macro-averaged | `max` |
| `jaccard_micro` | Jaccard index, micro-averaged | `max` |
| `jaccard_weighted` | Jaccard index, weighted by support | `max` |
| `hamming` | 1 − Hamming loss | `max` |
| `zero_one_loss` | Fraction of misclassified samples | `min` |
| `loss` | Training / validation loss | `min` |

### Multi-label metrics

All classification metrics above are available plus:

| Metric name | Description | Mode |
|---|---|---|
| `f1_samples` | F1 score, sample-averaged (default for multi-label) | `max` |
| `fbeta_<β>` | Fβ score (e.g. `fbeta_0.5`, `fbeta_2`), sample-averaged | `max` |
| `precision` | Precision, sample-averaged | `max` |
| `recall` | Recall, sample-averaged | `max` |
| `jaccard_samples` | Jaccard index, sample-averaged | `max` |

> **Tip:** For multi-label tasks, `fbeta_0.5` (precision-heavy) and `fbeta_2` (recall-heavy) are especially useful when you want to tune augmentations towards fewer false positives or fewer false negatives respectively.

### Custom metric example

```python
from bnnr import BNNRConfig, BNNRTrainer

# Use fbeta_0.5 as the metric driving augmentation selection
cfg = BNNRConfig(
    metrics=["accuracy", "fbeta_0.5", "f1_macro", "loss"],
    selection_metric="fbeta_0.5",
    selection_mode="max",
    # ... other fields ...
)

# Or pass a completely custom callable metric
def my_metric(preds, labels):
    return float((preds == labels).mean())

trainer = BNNRTrainer(
    model=adapter,
    train_loader=train_loader,
    val_loader=val_loader,
    augmentations=augmentations,
    config=cfg,
    custom_metrics={"my_metric": my_metric},
)
```

## Output and report fields

- `checkpoint_dir` (default: `checkpoints`)
- `report_dir` (default: `reports`)
- `report_preview_size` (default: `224`)
- `report_xai_size` (default: `512`)
- `dual_xai_report` (default: `false`)
- `report_probe_images_per_class` (default: `3`)
- `report_probe_max_classes` (default: `10`)

## XAI and cache fields

- `xai_enabled` (default: `true`)
- `xai_samples` (default: `4`)
- `xai_method` (`opticam`, `gradcam`, `craft`, `nmf`, `nmf_concepts`, `real_craft`; default: `opticam`)
- `xai_cache_dir` (default: `null`)
- `xai_cache_samples` (default: `0` = whole dataset)
- `xai_cache_max_samples` (default: `50000`)
- `xai_cache_force_recompute` (default: `false`)
- `xai_cache_progress` (default: `true`)
- `xai_selection_weight` (default: `0.0`, validated to `[0,1]`)
- `xai_pruning_threshold` (default: `0.0`, validated to `[0,1]`)
- `adaptive_icd_threshold` (default: `false`)

## Candidate pruning fields

- `candidate_pruning_enabled` (default: `true`)
- `candidate_pruning_relative_threshold` (default: `0.9`, validated `(0,1]`)
- `candidate_pruning_warmup_epochs` (default: `1`, validated `>0`)
- `reeval_baseline_per_iteration` (default: `false`)

## Event logging fields

- `event_log_enabled` (default: `true`)
- `event_sample_every_epochs` (default: `1`, validated `>0`)
- `event_xai_every_epochs` (default: `1`, validated `>0`)
- `event_min_interval_seconds` (default: `0.0`, validated `>=0`)

## Input denormalization fields

- `denormalization_mean` (default: `null`)
- `denormalization_std` (default: `null`)

## Task-specific fields

### `task: classification` (default)
No extra required fields.

### `task: multilabel`
- `multilabel_threshold` (default: `0.5`, validated to `(0,1)`)

Auto-default behavior in code for multilabel:
- If still at classification defaults, `selection_metric` becomes `f1_samples`
- If still at classification defaults, `metrics` becomes `[f1_samples, f1_macro, accuracy, loss]`

You can override these defaults to use any supported metric, for example:

```yaml
task: multilabel
selection_metric: fbeta_0.5
metrics: [fbeta_0.5, f1_samples, accuracy, loss]
```

**CLI note:** `python -m bnnr train` with built-in datasets does not construct multi-label loaders or `BCEWithLogitsLoss`; use the Python API or `examples/multilabel/` for end-to-end multi-label runs.

### `task: detection`

- `detection_bbox_format` (default: `xyxy`) — bounding box format, one of `xyxy`, `xywh`, `cxcywh`
- `detection_score_threshold` (default: `0.5`, validated to `[0, 1]`) — minimum confidence for evaluation predictions
- `detection_targets_mode` (default: `auto`) — augmentation target handling: `auto` (let BNNR decide), `image_only`, or `bbox_aware`
- `detection_class_names` (default: `null`) — optional list of class name strings for per-class reports

Auto-default behavior in code for detection:
- If still at classification defaults, `selection_metric` becomes `map_50`
- If still at classification defaults, `metrics` becomes `[map_50, map_50_95, loss]`

Additional detection fields for advanced use:
- `detection_nms_threshold` — NMS IoU threshold
- `detection_min_box_area` — minimum box area to keep
- `detection_max_truncation` — maximum box truncation ratio
- `detection_xai_method` — XAI method for detection (`saliency`, `occlusion`)
- `detection_xai_grid_size` — grid resolution for occlusion XAI
- `detection_xai_max_gt_boxes` — max ground-truth boxes rendered in XAI panels
- `detection_xai_max_pred_boxes` — max prediction boxes rendered in XAI panels

Use `DetectionAdapter` or `UltralyticsDetectionAdapter` as the model adapter. See [detection.md](detection.md) for the full detection guide.
