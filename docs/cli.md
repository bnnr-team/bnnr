# CLI Reference

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
Command reference for `python3 -m bnnr` aligned to `src/bnnr/cli.py`.

## When to use this page
Use this for no-code workflows: training, report reading, dashboard replay/export.

## Entry point

```bash
python3 -m bnnr --help
```

## `demo`

```bash
python3 -m bnnr demo
```

Zero-config onboarding run — **no flags, no YAML**.

- Downloads **CIFAR-10** automatically on first use.
- Trains a small **demo CNN** (not ResNet-18) with preset **`demo`** (saliency-guided **ICD** + ChurchNoise).
- Sample limits: 128 train / 64 val; `m_epochs=1`, `max_iterations=1`, XAI enabled.
- Starts the **live dashboard** and opens it in your browser (same as `--with-dashboard` on `train`).
- Writes artifacts under `reports/` and `checkpoints/`; after training (before the dashboard wait loop) prints paths to the JSON report and `reports/<run>/artifacts/xai/` heatmaps when present.

For an interactive wizard with prompts, use [`quickstart`](#quickstart) instead.

## `quickstart`

```bash
python3 -m bnnr quickstart [OPTIONS]
```

Interactive zero-config demo: prompts for dataset, preset, and dashboard, then runs training with built-in defaults and sample limits (128 train / 64 val).

Options: `--dashboard-port`, `--no-auto-open`.

## `train`

```bash
python3 -m bnnr train [OPTIONS]
```

Run BNNR augmentation search training.

### Options

- `--config -c` PATH (optional YAML config. Omit for built-in quickstart defaults)
- `--dataset` TEXT (dataset: `mnist`, `fashion_mnist`, `cifar10`, `stl10`, `imagefolder`, `coco_mini`, `yolo`) [default: cifar10]
- `--data-dir` PATH (directory for dataset download/storage) [default: data]
- `--data-path` PATH (custom data path (required for the `imagefolder`/`coco_mini`/`yolo`))
- `--output -o` PATH (output directory for checkpoints and reports)
- `--device -d` TEXT (device: `cuda`, `cpu`, `auto`)
- `--epochs -e` INTEGER (number of epochs per candidate)
- `--seed -s` INTEGER (random seed)
- `--no-xai` (disable XAI generation)
- `--augmentation-preset, --preset` TEXT (augmentation preset: `auto`, `light`, `standard`, `aggressive`, `gpu`, `icd`, `none`; unknown names fall back to `auto` with a warning). `icd` = saliency-guided ICD + AICD candidates (model/target layers supplied by the pipeline) [default: auto]
- `--with-dashboard / --without-dashboard` (enable dashboard: starts server, logs events, opens browser) [default: with-dashboard]
- `--dashboard-port` INTEGER (dashboard server port; if busy, the dashboard auto-falls back to the next free port in `port..port+9` and prints the actual port) [default: 8080]
- `--no-auto-open` (don't auto-open browser when dashboard starts)
- `--token` TEXT (token to protect dashboard control endpoints (pause/resume). Also configurable via `BNNR_DASHBOARD_TOKEN` env var)
- `--batch-size` INTEGER (training batch size) [default: 64]
- `--max-train-samples` INTEGER (limit training samples)
- `--max-val-samples` INTEGER (limit validation samples)
- `--num-classes` INTEGER (number of classes (for `imagefolder`))
- `--dry-run` (build the pipeline, print the summary + config warnings, then exit without training)
- `--help` (show this message and exit)

### Supported datasets

**Classification (built-in single-label demos):**

- `mnist`, `fashion_mnist`, `cifar10`, `stl10`, `imagefolder`

**Object detection (requires `task: detection` in your YAML and a matching config; see [detection.md](detection.md)):**

- `coco_mini` — COCO-style layout under `--data-path` (`train2017`/`val2017` or `train`/`val`, plus `annotations/`).
- `yolo` — Ultralytics-style `--data-path` pointing at `data.yaml` or its parent directory.

### Multi-label classification

`bnnr train` with **mnist**, **fashion_mnist**, **cifar10**, **stl10**, or **imagefolder** always builds **single-label** pipelines (`CrossEntropyLoss`, one class index per sample). Setting `task: multilabel` in your config YAML **does not** change that behavior. For multi-label, use the Python API ([golden_path.md](golden_path.md)) or the scripts under `examples/multilabel/` ([examples.md](examples.md)).

### Behavior notes

- `--with-dashboard` (default): starts live dashboard server and keeps process alive.
- `--without-dashboard`: no live server; good for one-shot runs.
- CLI keeps event logging enabled so `dashboard export` works after training.

### Examples

```bash
# Zero-config quickstart (built-in defaults)
python3 -m bnnr train --dataset cifar10 --preset light --with-dashboard

# Custom YAML config
python3 -m bnnr train \
  -c examples/configs/classification/mnist_example.yaml \
  --dataset mnist \
  --max-train-samples 1000 \
  -e 2

# CIFAR-10 with GPU augmentations
python3 -m bnnr train \
  -c examples/configs/classification/cifar10_example.yaml \
  --dataset cifar10 \
  --preset gpu \
  --device cuda
```

## `analyze`

```bash
python3 -m bnnr analyze [OPTIONS]
```

Run model analysis: metrics, XAI, data quality, failure patterns, recommendations.

On Windows terminals with encoding issues, set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` before running `bnnr analyze`.

See `analyze.md` for details and examples.

### Arguments (required)

- `--model -m` PATH (path to model checkpoint (`.pt`) or state dict)
- `--data` PATH (path to data directory (ImageFolder) or dataset name (e.g. `mnist`, `cifar10`))
- `--output -o` PATH (output directory for `analysis_report.json` and `report.html`)

### Options

- `--task -t` TEXT (task: classification or multilabel only (detection is not supported by analyze yet)) [default: classification]
- `--config -c` PATH (optional YAML config (overrides defaults))
- `--max-worst` INTEGER (number of worst predictions to include) [default: 20]
- `--no-xai` (disable XAI analysis)
- `--no-data-quality` (disable data quality checks)
- `--device -d` TEXT (device: cuda, cpu, auto)
- `--batch-size` INTEGER (batch size for evaluation) [default: 64]
- `--summary/--no-summary` (print executive summary and top findings/recommendations to stdout) [default: summary]
- `--cv-folds` INTEGER (optional number of folds for lightweight cross-validation (0 to disable)) [default: 0]
- `--xai-samples` INTEGER (number of samples for XAI probe set (more = more accurate, slower)) [default: 500]
- `--help` (show this message and exit)

## `report`

```bash
python3 -m bnnr report [OPTIONS]
```

View or export a BNNR training report.

### Arguments (required)

- `report_path` PATH

### Options

- `--format -f` TEXT (output format: `summary`, `json`) [default: summary]
- `--output -o` PATH
- `--help` (show this message and exit)

## Dashboard commands

```bash
python3 -m bnnr dashboard serve --run-dir reports --port 8080
python3 -m bnnr dashboard export --run-dir reports/run_YYYYMMDD_HHMMSS --output exported_dashboard
```

`dashboard serve` options:

- `--run-dir`
- `--port`
- `--frontend-dist`
- `--token` (or env `BNNR_DASHBOARD_TOKEN`)

`dashboard export` options:

- `--run-dir` (required)
- `--output` (required)
- `--frontend-dist`

## Dashboard usage notes (important)

- `dashboard serve` prints both Local URL and Network URL plus terminal QR code.
- Open Local URL on desktop first, then use QR from phone on the same network.
- For secured controls, use `--token` (or `BNNR_DASHBOARD_TOKEN`).
- For production workflow details (pause/resume, mobile access, export), see `dashboard.md`.

## Utility commands

```bash
python3 -m bnnr list-augmentations
python3 -m bnnr list-augmentations --verbose
python3 -m bnnr list-presets
python3 -m bnnr list-datasets
python3 -m bnnr version
```
