<p align="center">
  <img src="docs/assets/logo.png" alt="BNNR Logo" width="180">
</p>

<p align="center">
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/v/bnnr?v=1" alt="PyPI"></a>
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/pyversions/bnnr?v=1" alt="Python"></a>
  <a href="https://github.com/bnnr-team/bnnr/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bnnr-team/bnnr?v=1" alt="License"></a>
  <a href="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml"><img src="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

# BNNR (Bulletproof Neural Network Recipe)

BNNR is a PyTorch toolkit that makes vision models production-ready through a closed loop: train a model, explain its decisions with XAI, improve it with intelligent augmentation, and prove the result with structured reports and a live dashboard.

Supported tasks (**v0.2.1**):
- Single-label classification (`task: classification`)
- Multi-label classification (`task: multilabel`)
- Object detection (`task: detection`) — COCO-mini / YOLO pipelines; see `examples/detection/`

## Quickstart

### 1) Install

```bash
pip install bnnr
```

With optional extras:

```bash
pip install "bnnr[dashboard]"       # live dashboard (React + FastAPI)
pip install "bnnr[gpu]"             # GPU-accelerated augmentations (Kornia)
pip install "bnnr[albumentations]"  # Albumentations integration
```

For development from source:

```bash
git clone https://github.com/bnnr-team/bnnr.git
cd bnnr
# Dashboard UI must exist before editable install (hatch force-include).
(cd dashboard_web && npm ci && npm run build)
pip install -e ".[dev,dashboard]"
```

### Examples, notebooks, and markdown docs

The PyPI **wheel** ships the `bnnr` package only. **Runnable scripts** (`examples/`), **notebooks**, and the **documentation tree** (`docs/`) live in the [GitHub repository](https://github.com/bnnr-team/bnnr). After cloning, follow [docs/examples.md](docs/examples.md) (for example `PYTHONPATH=src python3 examples/...` from the repo root).

### 2) Create a minimal config

```bash
cat > /tmp/bnnr_quickstart.yaml <<'YAML'
m_epochs: 1
max_iterations: 1
metrics: [accuracy, f1_macro, loss]
selection_metric: accuracy
selection_mode: max
checkpoint_dir: checkpoints_quickstart
report_dir: reports_quickstart
xai_enabled: false
device: auto
seed: 42
candidate_pruning_enabled: false
YAML
```

### 3) Run a short training with live dashboard

```bash
python3 -m bnnr train \
  --config /tmp/bnnr_quickstart.yaml \
  --dataset cifar10 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --preset light \
  --with-dashboard \
  --dashboard-port 8080
```

Expected terminal markers:
- `BNNR PIPELINE SUMMARY`
- `BASELINE TRAINING`
- `TRAINING COMPLETE`
- `Report JSON    : reports_quickstart/run_.../report.json`
- `Dashboard      : http://127.0.0.1:8080/`

After training, keep the process running and open dashboard:

- desktop: `http://127.0.0.1:8080/`
- mobile: scan QR code printed in terminal (same Wi-Fi)

Stop dashboard with `Ctrl+C` when done.

### 4) Inspect the report

```bash
RUN_DIR=$(ls -1dt reports_quickstart/run_* | head -n 1)
python3 -m bnnr report "$RUN_DIR/report.json" --format summary
```

If you prefer one-shot (no live dashboard), run the same command with `--without-dashboard`.

## Main CLI commands

```bash
python3 -m bnnr --help
python3 -m bnnr train --help
python3 -m bnnr report --help
python3 -m bnnr list-datasets
python3 -m bnnr list-augmentations -v
python3 -m bnnr list-presets
python3 -m bnnr dashboard serve --run-dir reports --port 8080
python3 -m bnnr dashboard export --run-dir reports/run_YYYYMMDD_HHMMSS --out exported_dashboard
```

## Python API entry points

For custom models and dataloaders:
- `BNNRConfig`
- `BNNRTrainer`
- `SimpleTorchAdapter`
- `quick_run()`

See `docs/api_reference.md` and `docs/golden_path.md`.

## Repository docs

- `docs/README.md` (documentation index and source of truth)
- `docs/getting_started.md`
- `docs/dashboard.md`
- `docs/configuration.md`
- `docs/cli.md`
- `docs/api_reference.md`
- `docs/golden_path.md`
- `docs/detection.md`
- `docs/augmentations.md`
- `docs/examples.md`
- `docs/notebooks.md`
- `docs/artifacts.md`
- `docs/troubleshooting.md`

## Requirements

From `pyproject.toml`:
- Python `>=3.10`
- Core dependencies include `torch`, `torchvision`, `numpy`, `typer`, `pydantic`, `pyyaml`
- Dashboard extra adds `fastapi`, `uvicorn`, `websockets`, `qrcode`

## License

MIT — free and open source, forever. BNNR is an academic and community-driven project built for learning, research, and fun. No paid tiers, no commercial offerings — just open-source tools for the CV community.
