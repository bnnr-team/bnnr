# Configuration (`BNNRConfig`)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
All configuration fields currently implemented in `src/bnnr/config_model.py` (re-exported as `bnnr.BNNRConfig`), grouped by responsibility, with defaults and validation notes.

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
- `selector` (default: `metric_argmax`) — which rule picks the winning candidate. See below.
- `early_stopping_patience` (default: `2`)
- `device` (`cuda`, `cpu`, `auto`; default: `auto`)
- `seed` (default: `42`)
- `save_checkpoints` (default: `true`)
- `verbose` (default: `true`)
- `log_file` (default: `null`)

## Candidate selectors

`selector` names the rule that picks the winning candidate once every candidate has been evaluated.

| value | rule |
|---|---|
| `metric_argmax` (default) | greedy argmax on `selection_metric`, blended with XAI quality when `xai_selection_weight > 0` |
| `random` | uniform pick among the candidates, seeded from `seed` |
| `diagnosis` | picks the family the attention diagnosis recommends; see [diagnosis](diagnosis.md) |

Both are gated the same way: whatever a selector picks is discarded unless it beat the baseline on `selection_metric`. That gate is deliberately shared, so a comparison between two selectors is a comparison of their ranking rules and nothing else. A selector that skipped it would look better purely by accepting runs the others reject.

### Indistinguishability test

- `indistinguishable_resamples` (default: `2000`, validated `>= 100`)
- `indistinguishable_confidence` (default: `0.95`, validated in `(0, 1)`)

A candidate only replaces the baseline when the **paired bootstrap interval on the difference excludes zero** — not merely when its metric is a larger number.

T20's central negative result is that the selector was picking between candidates that are not distinguishable on the criterion it uses. On Waterbirds the candidate accuracies were .8749 / .8816 / .8549 on a validation set whose binomial standard error is larger than that spread. A strict `>` on those numbers is a coin flip.

When the interval covers zero, `SelectionResult.reason` becomes `"indistinguishable"` and **the baseline is kept**: it is the closest thing to what the data supports, and it is the cheapest in epochs. The interval lands in the run record so the decision is auditable afterwards rather than only at the moment it was made.

The comparison is **paired**: both arms are evaluated on the same validation set, and each bootstrap draw indexes both arms together. Ignoring the pairing would inflate the interval with variance that comes from the validation set rather than from the arms, and make everything look indistinguishable.

The statistic is the **mean** paired difference, not the median. At sample level the paired difference is in `{-1, 0, 1}`, where the median is degenerate; the mean of those differences *is* the accuracy difference. The seed-level convention in the benchmark summarizers is a median, because a per-seed metric is continuous — different level of aggregation, different right answer.

The test runs only when per-sample predictions were cached for both the candidate and the baseline. Without them the raw metric comparison stands, so a caller that never cached predictions behaves exactly as before.

### How the composite scales candidate differences

When `xai_selection_weight > 0` (deprecated, see below), the metric term is scaled by the **binomial standard error** `sqrt(p(1-p)/n)` of the selection metric, where `n` is the validation sample count.

It used to be min-max normalised across the candidates. That stretches whatever spread the candidates happened to have across the full `[0, 1]` range, so a spread of 0.2 pp and a spread of 20 pp both produce a metric term running 0 to 1, and the blending weight means something different in every iteration. On a saturated metric it was applied to scatter rather than to signal.

A normalised difference of `1.0` now means "one standard error better than the weakest candidate", in every iteration. A sub-noise spread produces a term well below 1; a spread of several standard errors exceeds 1 and dominates, which is correct because that difference is real.

The unit falls back to the observed spread when the sample size is unknown or the metric is not a proportion (a loss, say), since the binomial form claims nothing there.

**Standard deviation across the candidates is deliberately not used.** Three points is a poor estimator of anything, and on the runs this was written for it would have been an estimator of the very noise it was supposed to divide out.

### Deprecated XAI selection knobs

`xai_selection_weight` and `xai_pruning_threshold` are **deprecated as of 0.x**. Both keep working and both emit a `DeprecationWarning` when you set them to a non-zero value.

Both are built on `compute_xai_quality_score`, a hand-weighted scalar whose component weights were chosen by hand and never calibrated against an outcome. The T20 benchmark traced its null result to exactly this path.

There were two separate defects:

**Accuracy was counted twice.** The quality score carried accuracy at 25 %, and `select_best_path` then blended that score against the normalised selection metric — which is accuracy. The effective accuracy weight of the composite was a number nobody chose, and the two copies were not even the same quantity: one min-max normalised across candidates, the other the raw probe-set fraction. Accuracy is no longer part of the weighted sum; it remains in the breakdown as an observation.

**The score encodes a prior.** It rewards high Gini and penalises `edge_ratio` unconditionally, so it hard-codes "concentrated, central saliency is good". That is precisely the question [the diagnosis](diagnosis.md) answers per case — and for a Waterbirds-like model the built-in prior points the wrong way, since a model reading background context has diffuse, border-heavy attention and needs ICD, which this score ranks last.

**Preset change.** `xai_full` and `xai_adaptive` shipped `xai_selection_weight` of 0.1 and 0.15. Both are now `0.0`. Everything those presets are actually for — XAI reporting, dual XAI, adaptive ICD thresholds — is unchanged. Set the field yourself if you want the old behaviour.

`xai_pruning_threshold` keeps its preset values, since dropping a candidate is recoverable in a way that selecting the wrong one is not.

### Shadow mode

- `shadow_mode` (default: `true`)

Records the attention evidence on every run and lets none of it influence selection. For each candidate of each iteration it writes the saliency statistics, the robustness metrics, and whether that candidate was the one the run kept, to `shadow_records.jsonl` in the run directory.

It is on by default because it costs nothing beyond maps the run already produces. It does nothing when `xai_enabled` is `false`, since there are no maps to read.

**Records are raw statistics, never a regime.** Writing a regime would need thresholds, and the thresholds are exactly what these records exist to calibrate. It needs no `diagnosis:` block and guesses nothing.

**Every candidate is recorded, not only the winner.** A calibration set containing only winning arms cannot answer "would the other choice have been better", which is the question the whole exercise turns on.

`perturbation_shift` is deliberately not computed here: it needs a second explainer pass, and shadow mode's claim is that it is free. The three statistics that come from maps the run already made are what get recorded.

### Diagnosis thresholds

`selector: diagnosis` requires a `diagnosis:` block, and **every field in it starts unset**:

```yaml
selector: diagnosis
diagnosis:
  concentration_lo: 0.31
  concentration_hi: 0.58
  border_mass_hi: 0.34
  perturbation_shift_hi: 0.47
  robustness_gap_hi: 0.12
  min_confidence: 0.75        # optional
```

Asking for the selector with any of the first five unset raises at **config construction**, not mid-run, so the failure lands before any GPU time is spent. There is deliberately no default: shipping one would repeat the mistake that produced `xai_selection_weight` and its preset 0.1 and 0.15.

Thresholds are calibrated per model family and per saliency resolution, so they travel as a file rather than as library defaults:

```python
from bnnr.config import load_diagnosis_profile

thresholds = load_diagnosis_profile(Path("profiles.yaml"), "imagewoof_resnet50")
config = BNNRConfig(selector="diagnosis", diagnosis=thresholds)
```

A profile file holds one or more named sets. With several in the file the name is required — picking one silently would be the same class of mistake as a default threshold.

Setting the block without setting `selector` is harmless and is what shadow mode does: it records the raw statistics and needs no thresholds at all.

`hard_quantile_q` is **not** part of this block. It lives at the top level because `hard_quantile_acc` and `robustness_gap` are worth watching with no diagnosis configured; the calibration study sweeps it alongside these thresholds, which does not make it one.

`diagnosis` needs calibrated thresholds and a `Diagnosis` for the iteration. Without one it selects nothing rather than falling back to argmax, because a silent fallback would make a benchmark contrast between the two measure a blend of them.

`random` is not a joke setting. It is the arm the T20 benchmark ran `metric_argmax` against, and `metric_argmax` did not beat it at n=10 across two datasets; having it as a named selector is what makes that contrast reproducible rather than something the benchmark harness improvises.

Changing `selector` does not change what `select_best_path()` is called or how — the function is now a thin adapter over the registry, so existing code picks up the setting without modification.

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
The cache is keyed by sample index, so the DataLoader has to yield `(image, label, index)`. Wrap the dataset with `bnnr.IndexedDataset` if it does not. Without indices no map can be persisted and saliency is recomputed every batch; BNNR warns once per run when that happens.

- `xai_cache_dir` (default: `null`): when `null`, the cache lives under the current run directory (`<report_dir>/run_<timestamp>/xai_cache`), so saliency maps are never silently reused across runs. Set an explicit path to share a cache between runs (you own invalidation in that case).
- `xai_cache_samples` (default: `0` = whole dataset)
- `xai_cache_max_samples` (default: `50000`)
- `xai_cache_max_mb` (default: `2048`, validated `>=0`): disk cap for the on-disk XAI cache, in megabytes. After precompute the cache is trimmed LRU-by-mtime (oldest maps evicted first) until it fits under this. `0` disables the cap. Only index-keyed precompute maps are persisted, so the cache is naturally bounded by the dataset size; this cap is a disk-budget safety net.
- `xai_cache_force_recompute` (default: `false`)
- `xai_cache_progress` (default: `true`)

The XAI cache is precomputed **after** the baseline phase, so masks from `ICD`/`AICD` are guided by the trained baseline model rather than random initial weights, and is computed once for all branch-search iterations. A `manifest.json` records the XAI method, dataset size, image shape, and a fingerprint of the baseline model's weights; if any of these differs from the cached maps (for example a different `xai_method`, or a **different model** sharing an explicit `xai_cache_dir`), the stale maps are dropped and recomputed.
- `xai_selection_weight` (default: `0.0`, validated to `[0,1]`)
- `xai_pruning_threshold` (default: `0.0`, validated to `[0,1]`)
- `adaptive_icd_threshold` (default: `false`)
- `duplicate_hamming_threshold` (default: `10`, validated `>=0`) — Hamming distance threshold for duplicate-sample detection in XAI cache

## Candidate pruning fields

- `candidate_pruning_enabled` (default: `true`)
- `candidate_pruning_relative_threshold` (default: `0.9`, validated `(0,1]`)
- `candidate_pruning_warmup_epochs` (default: `1`, validated `>0`)
- `reeval_baseline_per_iteration` (default: `false`)

## Hard-quantile robustness fields

- `hard_quantile_q` (default: `0.2`, validated `(0,1]`)

The fraction of the validation set treated as "hard", ranked by per-sample loss. Every evaluation that caches predictions adds three fields to its metrics:

| field | meaning |
|---|---|
| `hard_quantile_acc` | accuracy restricted to the highest-loss `hard_quantile_q` fraction |
| `robustness_gap` | `overall accuracy - hard_quantile_acc` |
| `hard_quantile_q` | the `q` those two were computed with |

This is a label-free stand-in for "poor robustness to context shift". Group labels would answer the question directly, but consuming them costs the assumption BNNR is built on: images and labels, nothing else. Inferring the hard group from the loss is what the JTT/EIIL family does instead.

A model that is uniformly mediocre has a small gap. A model that is excellent on the majority and fails a minority has a large one, which is the shape of a shortcut. The attention diagnosis reads `robustness_gap`; on its own it is a diagnostic you can watch.

The loss is plain cross-entropy on the logits the prediction cache already captures, so it costs no second pass over the loader. It is deliberately not the trainer's own criterion: a weighted or label-smoothed criterion ranks samples by class frequency as much as by difficulty, and the ranking is the entire point. The three fields are single-label classification only; multilabel and detection runs do not carry them.

## Event logging fields

- `event_log_enabled` (default: `true`)
- `event_sample_every_epochs` (default: `1`, validated `>0`)
- `event_xai_every_epochs` (default: `1`, validated `>0`)
- `event_min_interval_seconds` (default: `0.0`, validated `>=0`)

## Input denormalization fields

- `denormalization_mean` (default: `null`)
- `denormalization_std` (default: `null`)

BNNR augmentations operate on unnormalised images, so a batch that has already
been through `transforms.Normalize()` has to be converted back before it can be
augmented. Set both fields to the statistics your DataLoader used and BNNR
undoes the normalisation before each augmentation and reapplies it afterwards.
The values are also used for report previews and XAI overlays.

If a batch arrives outside both `[0, 1]` and `[0, 255]` and these fields are not
set, BNNR raises `NormalisedInputError` rather than clipping the batch into
`[0, 1]`, which would destroy the image without any visible error. The two ways
out are to remove `Normalize()` from the DataLoader transforms and rely on
BatchNorm in the model, which is what the built-in pipelines do, or to set these
two fields.

Batches that are already in `[0, 1]` or `[0, 255]` are never denormalised, so
setting these fields for reporting alone does not change how such a batch is
augmented.

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
- `detection_xai_method` (default: `activation`) — XAI method for detection: `activation` (backbone activation heatmap) or `occlusion`
- `detection_xai_grid_size` — grid resolution for occlusion XAI
- `detection_xai_max_gt_boxes` — max ground-truth boxes rendered in XAI panels
- `detection_xai_max_pred_boxes` — max prediction boxes rendered in XAI panels

Use `DetectionAdapter` or `UltralyticsDetectionAdapter` as the model adapter. See [detection.md](detection.md) for the full detection guide.
