# BNNR Documentation

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
A code-verified documentation index for installation, configuration, CLI/API usage, outputs, troubleshooting, and development workflows.

## When to use this page
Use this as the starting point when you need the shortest path to the page relevant to your task.

- `getting_started.md` — first run from a clean machine
- `configuration.md` — all `BNNRConfig` fields and defaults from `src/bnnr/core.py`
- `cli.md` — CLI commands and option reference from `src/bnnr/cli.py`
- `dashboard.md` — live/replay/mobile/QR dashboard workflow
- `api_reference.md` — public Python API usage (classification helpers and a short Detection / Ultralytics subsection)
- `analyze.md` — standalone model analysis (`bnnr analyze`): metrics, XAI, data quality, failure analysis
- `golden_path.md` — integrating BNNR with your own model and dataloaders
- `plugin_icd.md` — ICD/AICD in your own PyTorch loop (no BNNRTrainer; built on pytorch-grad-cam)
- `integrations.md` — pytorch-grad-cam and Ultralytics YOLO integration hub + stable URLs
- `augmentations.md` — presets and augmentation classes available in code
- `detection.md` — object detection guide (adapters, augmentations, config, metrics, XAI)
- `examples.md` — production usage guide for Python scripts under `examples/` (by subdirectory)
- `notebooks.md` — notebook execution and validation guide
- `artifacts.md` — on-disk run outputs (`report.json`, `events.jsonl`, artifacts)
- `troubleshooting.md` — real failure modes and fixes
- `development.md` — tests, linting, type checks, dashboard frontend build
- `benchmarks.md` — CIFAR-10 benchmark (no BNNR vs RandAugment vs BNNR branch search)
