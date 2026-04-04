# Model analysis (`bnnr analyze`)

## What you will find here

How to run **zero-friction diagnostics** on a trained model without running full BNNR training: metrics, XAI, data quality, failure analysis, patterns, and recommendations.

## When to use this page

Use this when you have a trained model (e.g. from `bnnr train` or your own pipeline) and want a single report that answers: how does the model perform, where does it fail, what does it look at (XAI), and what to improve.

## Classification quick start

For **image classification** (MNIST, CIFAR-10, STL-10, ImageFolder):

```bash
# 1. Train a model (or use an existing checkpoint)
python3 -m bnnr train -c config.yaml --dataset mnist -o out_mnist -e 2

# 2. Run analysis
python3 -m bnnr analyze --model out_mnist/checkpoints/iter_2_baseline.pt \
  --data mnist --output out_analyze

# 3. Open the HTML report
xdg-open out_analyze/report.html
```

The report includes: accuracy/F1, per-class diagnostics, confusion matrix, XAI quality score, failure patterns, and actionable recommendations. API and CLI produce equivalent results for the same model and data.

## Overview

`bnnr analyze` (and the Python API `analyze_model`) runs:

1. **Evaluation** — forward pass on the validation loader → metrics (e.g. accuracy, F1), per-class accuracy, confusion matrix.
2. **XAI** (optional) — saliency maps and rich analysis (focus, edge ratio, quality score, class diagnoses) on a probe set.
3. **Data quality** (optional) — duplicate detection, image quality checks on the validation data.
4. **Failure analysis** — per-sample predictions with loss/confidence; top-N worst predictions.
5. **Failure patterns** — e.g. top confused class pairs, classes with low XAI quality.
6. **Recommendations** — text list of improvement hints (e.g. add data for class X, consider ICD).

Output: `AnalysisReport` in memory, plus on disk: `analysis_report.json`, optional `report.html`, and artifact directories (e.g. `worst_overlays/` if XAI overlays are generated).

## CLI: `bnnr analyze`

```bash
python3 -m bnnr analyze --model PATH --data PATH_OR_DATASET --output DIR [OPTIONS]
```

### Required arguments

- `--model` — Path to a saved model checkpoint (`.pt`). Can be a BNNR checkpoint (with `model` or `model_state`) or a raw `state_dict`.
- `--data` — Either a directory path (ImageFolder-style: `class1/`, `class2/`, …) or a built-in dataset name: `mnist`, `fashion_mnist`, `cifar10`, `stl10`.
- `--output` — Directory where to write `analysis_report.json` and `report.html`.

### Main options

- `--task` — `classification` (default) or `multilabel` only. **Detection is not supported** by `analyze` until the main BNNR stack ships detection training end-to-end.
- `--config` — Optional path to a BNNR config YAML (for device, metrics, num_classes, etc.).
- `--max-worst` — Number of worst predictions to keep (default: 20).
- `--no-xai` — Disable XAI analysis.
- `--no-data-quality` — Disable data quality checks.
- `--device` — `cuda`, `cpu`, or `auto`.
- `--batch-size` — Batch size for evaluation (default from config or pipeline).
- `--cv-folds` — Optional number of folds for lightweight cross-validation on the validation set (`classification`: stratified single-label; `multilabel`: macro F1 / subset accuracy per fold; 0 disables).
- `--xai-samples` — Number of samples for XAI probe set (default: 500; more = more accurate, slower).

### Behavior notes

- The CLI builds a pipeline (dataset + adapter) from `--data` and loads the checkpoint into the adapter. For ImageFolder, use `--data /path/to/val_root`; the pipeline expects `--config` or compatible defaults (e.g. `num_classes` for imagefolder).
- XAI requires an adapter that implements `XAICapableModel` (e.g. `SimpleTorchAdapter` with `target_layers`). If the checkpoint was saved by BNNR train, the same config/dataset usually provides the right adapter.
- Output layout: `output_dir/analysis_report.json`, `output_dir/report.html`, and optionally `output_dir/worst_overlays/`, `output_dir/data_quality/` (see `artifacts.md`).

### Examples

```bash
# MNIST: checkpoint from a previous BNNR run
python3 -m bnnr analyze \
  --model reports/run_20250101_120000/checkpoints/best.pt \
  --data mnist \
  --output ./analysis_out

# ImageFolder with config
python3 -m bnnr analyze \
  --model ./my_model.pt \
  --data /path/to/validation_images \
  --output ./analysis_out \
  --config my_config.yaml

# No XAI, no data quality (faster)
python3 -m bnnr analyze \
  --model best.pt \
  --data cifar10 \
  --output ./out \
  --no-xai \
  --no-data-quality
```

## Python API

### `analyze_model`

```python
from bnnr import analyze_model, SimpleTorchAdapter
import torch

# adapter: ModelAdapter with your model (and target_layers for XAI)
# val_loader: DataLoader yielding (image, label, index) for classification
report = analyze_model(
    adapter,
    val_loader,
    config=None,  # optional BNNRConfig; defaults for task/device/metrics
    task="classification",
    device="auto",
    output_dir="./analysis_out",
    run_data_quality=True,
    max_worst=20,
    xai_method="opticam",
    xai_enabled=True,
    data_quality_max_samples=5000,
    cv_folds=3,  # optional lightweight k-fold CV
)

# In-memory report
print(report.metrics)
print(report.per_class_accuracy)
print(report.failure_patterns_list())
print(report.recommendations)

# Save to disk
report.save("./analysis_out")
report.to_html("./analysis_out/report.html")
```

### `AnalysisReport`

- **Attributes (core)**: `metrics`, `per_class_accuracy`, `confusion`, `xai_insights`, `xai_diagnoses`, `xai_quality_summary`, `data_quality_result`, `failure_patterns`, `recommendations`.
- **Attributes (extended v0.2)**:
  - `schema_version` — report schema version string.
  - `executive_summary` — health badge/score, key findings, top actions, critical classes.
  - `findings` — structured, root-cause oriented findings (type, evidence, interpretation, severity).
  - `recommendations_structured` — prioritized, causal recommendations linked to findings.
  - `class_diagnostics` — per-class precision/recall/F1/support/severity.
  - `true_distribution`, `pred_distribution`, `distribution_summary` — class distribution and collapse/bias hints.
  - `failure_patterns_extended` — enriched failure taxonomy (zero/near-zero recall, collapse, dominant bias, XAI-based patterns, etc.).
  - `xai_quality_per_class`, `xai_examples_per_class` — per-class XAI quality and sample overlays.
  - `data_quality_summary` — dataset health summary (scanned samples, duplicates, flagged images, warnings).
  - `cv_results` — lightweight k-fold CV results (if enabled).
  - `cluster_views` — 2D cluster visualisations of confusing examples (e.g. worst predictions).
- **Methods**:
  - `save(output_dir)` — writes `analysis_report.json` and optional artifact dirs.
  - `to_html(path)` — writes a single HTML report file.
  - `failure_patterns_list()` — returns the list of failure patterns.

For full API details, see `api_reference.md`.

## Limitations (v0.2.x)

- **Detection**: Not supported by `bnnr analyze` or `analyze_model`; use classification or multi-label only.
- **Compare**: No built-in comparison of two models (e.g. pre vs post fine-tuning); planned for a later release.
- **Events**: Analyze does not emit events to `events.jsonl`; it produces standalone artifacts only.

## See also

- `artifacts.md` — layout of `analysis_report.json` and analyze output directories.
- `api_reference.md` — `analyze_model`, `AnalysisReport`.
- `cli.md` — full CLI reference including `analyze` options.
