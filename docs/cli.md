# CLI Reference

## What you will find here
Command reference for `python3 -m bnnr` aligned to `src/bnnr/cli.py`.

## When to use this page
Use this for no-code workflows: training, report reading, dashboard replay/export.

## Entry point

```bash
python3 -m bnnr --help
```

## `train`

```bash
python3 -m bnnr train --config CONFIG [OPTIONS]
```

### Supported datasets

- `mnist`
- `fashion_mnist`
- `cifar10`
- `imagefolder`

### Multi-label classification

`bnnr train` with any of the datasets above always builds **single-label** pipelines (`CrossEntropyLoss`, one class index per sample). Setting `task: multilabel` in your config YAML **does not** change that behavior. For multi-label, use the Python API ([golden_path.md](golden_path.md)) or the scripts under `examples/multilabel/` ([examples.md](examples.md)).

### Main options

- `--config, -c` (required)
- `--dataset`
- `--data-dir`
- `--data-path` (required for `imagefolder`)
- `--output, -o`
- `--device, -d` (`cuda`, `cpu`, `auto`)
- `--epochs, -e`
- `--seed, -s`
- `--no-xai`
- `--augmentation-preset, --preset` (`auto`, `light`, `standard`, `aggressive`, `gpu`)
- `--with-dashboard / --without-dashboard`
- `--dashboard-port`
- `--no-auto-open`
- `--dashboard-token`
- `--batch-size`
- `--max-train-samples`
- `--max-val-samples`
- `--num-classes` (for `imagefolder`)

### Behavior notes

- `--with-dashboard` (default): starts live dashboard server and keeps process alive.
- `--without-dashboard`: no live server; good for one-shot runs.
- CLI keeps event logging enabled so `dashboard export` works after training.

### Examples

```bash
# CIFAR-10 one-shot run
python3 -m bnnr train \
  --config examples/configs/classification/cifar10_example.yaml \
  --dataset cifar10 \
  --preset light \
  --without-dashboard

# ImageFolder
python3 -m bnnr train \
  --config examples/configs/classification/imagefolder_example.yaml \
  --dataset imagefolder \
  --data-path /path/to/dataset
```

## `analyze`

```bash
python3 -m bnnr analyze --model PATH --data PATH_OR_DATASET --output DIR [OPTIONS]
```

Run standalone diagnostics on a trained model (no training): metrics, XAI, data quality, failure analysis, patterns, recommendations. Writes `analysis_report.json` and `report.html` under `--output`.

Required: `--model` (checkpoint `.pt`), `--data` (directory or dataset name: `mnist`, `fashion_mnist`, `cifar10`, `stl10`), `--output`.

Options: `--task` (classification or multilabel only; detection is not supported by analyze), `--config`, `--max-worst`, `--no-xai`, `--no-data-quality`, `--device`, `--batch-size`, `--cv-folds`, `--xai-samples`, `--summary/--no-summary`.

See `analyze.md` for details and examples.

## `report`

```bash
python3 -m bnnr report path/to/report.json --format summary
python3 -m bnnr report path/to/report.json --format json
python3 -m bnnr report path/to/report.json --format json --output report_payload.json
```

Notes:

- `--format html` is intentionally rejected in current CLI.
- `--output` writes rendered report output to file (summary text or JSON payload).
- Use dashboard export for static HTML output.

## Dashboard commands

```bash
python3 -m bnnr dashboard serve --run-dir reports --port 8080
python3 -m bnnr dashboard export --run-dir reports/run_YYYYMMDD_HHMMSS --out exported_dashboard
```

`dashboard serve` options:

- `--run-dir`
- `--port`
- `--frontend-dist`
- `--token` (or env `BNNR_DASHBOARD_TOKEN`)

`dashboard export` options:

- `--run-dir` (required)
- `--out` (required)
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
