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

**Classification (built-in single-label demos):**

- `mnist`
- `fashion_mnist`
- `cifar10`
- `imagefolder`

**Object detection (requires `task: detection` in your YAML and a matching config; see [detection.md](detection.md)):**

- `coco_mini` — COCO-style layout under `--data-path` (`train2017`/`val2017` or `train`/`val`, plus `annotations/`).
- `yolo` — Ultralytics-style `--data-path` pointing at `data.yaml` or its parent directory.

### Multi-label classification

`bnnr train` with **mnist**, **fashion_mnist**, **cifar10**, or **imagefolder** always builds **single-label** pipelines (`CrossEntropyLoss`, one class index per sample). Setting `task: multilabel` in your config YAML **does not** change that behavior. For multi-label, use the Python API ([golden_path.md](golden_path.md)) or the scripts under `examples/multilabel/` ([examples.md](examples.md)).

### Main options

- `--config, -c` (required)
- `--dataset`
- `--data-dir`
- `--data-path` (required for `imagefolder`, `coco_mini`, and `yolo`)
- `--output, -o`
- `--device, -d` (`cuda`, `cpu`, `auto`)
- `--epochs, -e`
- `--seed, -s`
- `--no-xai`
- `--augmentation-preset, --preset` (`auto`, `light`, `standard`, `aggressive`, `gpu`, `none`; unknown names fall back to `auto` with a warning)
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
