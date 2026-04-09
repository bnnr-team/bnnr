# BNNR Documentation

## What you will find here
A code-verified documentation index for installation, configuration, CLI/API usage, outputs, troubleshooting, and development workflows.

## When to use this page
Use this as the starting point when you need the shortest path to the page relevant to your task.

- `getting_started.md` — first run from a clean machine
- `configuration.md` — all `BNNRConfig` fields and defaults from `src/bnnr/core.py`
- `cli.md` — CLI commands and option reference from `src/bnnr/cli.py`
- `dashboard.md` — live/replay/mobile/QR dashboard workflow
- `api_reference.md` — public Python API usage (classification helpers and a short Detection / Ultralytics subsection)
- `golden_path.md` — integrating BNNR with your own model and dataloaders
- `augmentations.md` — presets and augmentation classes available in code
- `detection.md` — object detection guide (adapters, augmentations, config, metrics, XAI)
- `examples.md` — production usage guide for Python scripts under `examples/` (by subdirectory)
- `notebooks.md` — notebook execution and validation guide
- `artifacts.md` — on-disk run outputs (`report.json`, `events.jsonl`, artifacts)
- `troubleshooting.md` — real failure modes and fixes
- `development.md` — tests, linting, type checks, dashboard frontend build
