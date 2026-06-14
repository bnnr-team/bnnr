# Model analysis (`bnnr analyze`)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here

How to run **zero-friction diagnostics** on a trained model without running full BNNR training: metrics, XAI, data quality, failure analysis, patterns, and recommendations.

**Preview (live in browser, no install):** [sample HTML report on raw.githack.com](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html) — MNIST checkpoint, self-contained report.

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

## Torchvision pretrained checkpoint

Use this when you already have a **torchvision** classification model (e.g. ResNet on CIFAR-10) and want a failure/XAI report **without** running full BNNR training.

**CLI note:** `bnnr analyze --data cifar10` builds the built-in demo CNN (`_CifarCNN`), not ResNet-18. For torchvision architectures, use the Python API (below) or the runnable example.

**Runnable example (ResNet-18 + CIFAR-10 val set)**

```bash
PYTHONPATH=src python3 examples/classification/torchvision_analyze_cifar10.py --quick
xdg-open torchvision_analyze_out/report.html
```

With your own weights: train or export a `state_dict`, then pass `--checkpoint ./resnet18_cifar10.pt`. See [examples.md](examples.md) for smoke flags.

**CLI when the checkpoint matches the built-in pipeline**

After `python -m bnnr train --dataset cifar10`, analyze the saved demo-CNN checkpoint:

```bash
python3 -m bnnr analyze \
  --model ./checkpoints/iter_2_baseline.pt \
  --data cifar10 \
  --output ./out_analyze
```

For custom wrappers or non-BNNR checkpoints, see [golden_path.md](golden_path.md). Expected report layout: [sample HTML on raw.githack.com](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html).

**Note:** Object detection checkpoints are not supported by `bnnr analyze` yet (classification and multilabel only).

## Overview

`bnnr analyze` (and the Python API `analyze_model`) runs:

1. **Evaluation** — forward pass on the validation loader → metrics (e.g. accuracy, F1), per-class accuracy, confusion matrix.
2. **XAI** (optional) — saliency maps and rich analysis (focus, edge ratio, quality score, class diagnoses) on a probe set.
3. **Data quality** (optional) — duplicate detection, image quality checks on the validation data.
4. **Failure analysis** — per-sample predictions with loss/confidence; top-N worst predictions.
5. **Failure patterns** — e.g. top confused class pairs, classes with low XAI quality.
6. **Recommendations** — text list of improvement hints (e.g. add data for class X, consider ICD).

Output: `AnalysisReport` in memory, plus on disk: `analysis_report.json`, `report.html` (self-contained with embedded XAI PNGs when generated via CLI), and optional `artifacts/` for raw overlay files referenced in JSON.

## Portable HTML report

`bnnr analyze` writes **`report.html` as a single portable file**: confusion-pair saliency maps, best/worst examples, and data-quality thumbnails are embedded as **base64 data URIs**. You can email the file, open it from GitHub, or share it without the `artifacts/` folder.

- **CLI:** automatic (`artifact_root` = `--output` directory).
- **Python API:**

```python
report.to_html("./out/report.html", artifact_root="./out", embed_images=True)
```

`analysis_report.json` still stores relative paths to PNGs under `artifacts/` for tooling; only the HTML is self-contained.

Regenerate the public sample: `python scripts/generate_analyze_sample_report.py` → `docs/assets/analyze-report-sample.html`.

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
- `--summary/--no-summary` — Print (or suppress) executive summary, key findings, and top actions to stdout (default: enabled).

### Behavior notes

- The CLI builds a pipeline (dataset + adapter) from `--data` and loads the checkpoint into the adapter. For ImageFolder, use `--data /path/to/val_root`; the pipeline expects `--config` or compatible defaults (e.g. `num_classes` for imagefolder).
- XAI requires an adapter that implements `XAICapableModel` (e.g. `SimpleTorchAdapter` with `target_layers`). If the checkpoint was saved by BNNR train, the same config/dataset usually provides the right adapter.
- Output layout: `output_dir/analysis_report.json`, `output_dir/report.html` (portable, embedded images), and optionally `output_dir/artifacts/` (raw PNGs; see `artifacts.md`).
- `--cv-folds` is a lightweight estimate of metric variability on validation predictions: analyze runs one inference pass, then computes k-fold metrics on cached predictions (it does not train k separate models).

### Validation loader contract

`analyze` pairs each prediction with its source image by index so XAI overlays and per-sample confidences line up with the right picture. There are two ways the index is determined:

- **Indexed dataset (recommended):** the loader yields `(image, label, index)` 3-tuples, and the explicit `index` is used directly. This is robust to any sampling or shuffling.
- **2-tuple fallback:** the loader yields `(image, label)`, and analyze synthesizes the index as a running offset, which assumes iteration order equals dataset order.

Because the 2-tuple fallback breaks under shuffling, analyze automatically rebuilds a shuffled (`RandomSampler`) `DataLoader` with `shuffle=False` for the analysis pass and emits a warning. Order does not affect analyze (metrics are aggregated, XAI is keyed per index), so this only fixes mispairing. Pass an unshuffled loader (or an indexed 3-tuple dataset) to avoid the warning.

### Metric definitions (practical)

- `accuracy`, `precision`, `recall`, `f1`, `cohen_kappa`, and confusion-based diagnostics are computed in the evaluation/analysis pipeline (`run_evaluation` and downstream analysis modules).
- `ECE (top-1)` means expected calibration error computed on top-1 confidence bins.
- `Cohen's kappa` is agreement beyond chance (chance-corrected agreement), reported as a scalar in `[-1, 1]`.

### Windows UTF-8 note

If report generation logs look garbled on Windows terminals, run with UTF-8 enabled:

```bash
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -m bnnr analyze --model ... --data ... --output ...
```

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
  - `schema_version` — report schema version string (from 0.4.8 onward, matches the installed `bnnr` package version; older reports may show `0.2.1`).
  - `executive_summary` — health badge/score, key findings, top actions, critical classes.
  - `findings` — structured, root-cause oriented findings (type, evidence, interpretation, severity).
  - `recommendations_structured` — prioritized, causal recommendations linked to findings.
  - `class_diagnostics` — per-class precision/recall/F1/support/severity.
  - `true_distribution`, `pred_distribution`, `distribution_summary` — class distribution and collapse/bias hints.
  - `failure_patterns_extended` — enriched failure taxonomy (zero/near-zero recall, collapse, dominant bias, XAI-based patterns, etc.).
  - `xai_quality_per_class`, `xai_examples_per_class` — per-class XAI quality and sample overlays.
  - `data_quality_summary` — dataset health summary (scanned samples, duplicates, flagged images, warnings).
  - `cv_results` — lightweight k-fold CV results (if enabled).
  - `cluster_views` — reserved key in docs history; not emitted by current `analyze` implementation.
- **Methods**:
  - `save(output_dir)` — writes `analysis_report.json` and optional artifact dirs.
  - `to_html(path)` — writes a single HTML report file.
  - `failure_patterns_list()` — returns the list of failure patterns.

For full API details, see `api_reference.md`.

## Limitations (current code)

- **Detection**: Not supported by `bnnr analyze` or `analyze_model`; supported tasks are `classification` and `multilabel` only.
- **Compare**: `compare_runs` compares training `report.json` files; there is no built-in side-by-side compare of two `analyze` HTML reports in the CLI.
- **Events**: Analyze does not emit events to `events.jsonl`; it produces standalone artifacts only.
- **ROC/PR curves**: Analyze focuses on point metrics and diagnostics; ROC-AUC / PR curves are not rendered in `report.html`.
- **Advanced concept XAI (e.g. CRAFT/NMF)**: `analyze_model` uses saliency/CAM-style methods (`xai_method`, default `opticam`); CRAFT/NMF are available in training/XAI modules but not wired into the analyze pipeline.

## See also

- `artifacts.md` — layout of `analysis_report.json` and analyze output directories.
- `api_reference.md` — `analyze_model`, `AnalysisReport`.
- `cli.md` — full CLI reference including `analyze` options.
