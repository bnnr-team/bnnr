# Getting Started

## What you will find here
A step-by-step path for first-time users:
- installation,
- first training run,
- live dashboard (desktop + mobile via QR),
- report inspection,
- static export.

This page is written for junior ML users and assumes no prior BNNR knowledge.

## When to use this page
Use this on a fresh machine or first setup.

## 1) Requirements

- Python `>=3.9`
- `pip`
- Optional GPU support depends on your local PyTorch/CUDA installation

## 2) Create a clean venv and install

```bash
python3 -m venv /tmp/bnnr-venv
source /tmp/bnnr-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard]"
```

If `python3 -m venv` fails with `ensurepip is not available`, install your OS venv package (for example `python3.12-venv` on Ubuntu) and retry.
If you see `externally-managed-environment`, you're using system Python directly — activate the venv and run install commands inside it.

Optional extras:

```bash
python -m pip install -e ".[gpu]"
python -m pip install -e ".[albumentations]"
```

## 3) Sanity check CLI availability

```bash
python -m bnnr --help
python -m bnnr train --help
python -m bnnr dashboard serve --help
```

## 4) Create a minimal config

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

## 5) First run in live dashboard mode (recommended)

Run with dashboard enabled (`--with-dashboard` is default):

```bash
python -m bnnr train \
  --config /tmp/bnnr_quickstart.yaml \
  --dataset cifar10 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --preset light \
  --with-dashboard \
  --dashboard-port 8080 \
  --no-auto-open
```

What you should see in terminal:

- `BNNR PIPELINE SUMMARY`
- `BASELINE TRAINING`
- `TRAINING COMPLETE`
- `Report JSON    : .../report.json`
- `Dashboard      : http://127.0.0.1:8080/`

Important: in live dashboard mode, process stays alive after training to keep server running. Stop with `Ctrl+C` after your checks.

## 6) Open dashboard on desktop and mobile

### Desktop

Open:

- `http://127.0.0.1:8080/` (local browser)

### Mobile (same Wi-Fi)

1. In terminal, find `Network URL` and QR code.
2. Connect phone to the same network as your machine.
3. Scan QR code in terminal.
4. Open the URL on phone.

If QR/mobile does not work, see `troubleshooting.md` section for network blockers.

## 7) Protect dashboard controls (recommended)

For shared/dev-network runs, protect pause/resume endpoints with token:

```bash
python -m bnnr train \
  --config /tmp/bnnr_quickstart.yaml \
  --dataset cifar10 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --preset light \
  --with-dashboard \
  --dashboard-token "change-me"
```

Equivalent replay mode protection:

```bash
python -m bnnr dashboard serve --run-dir reports_quickstart --port 8080 --token "change-me"
```

## 8) Read the generated report

```bash
RUN_DIR=$(ls -1dt reports_quickstart/run_* | head -n 1)
python -m bnnr report "$RUN_DIR/report.json" --format summary
```

## 9) Replay dashboard for an existing run

```bash
python -m bnnr dashboard serve --run-dir reports_quickstart --port 8080
```

Use replay mode when:
- training is already finished,
- you want to inspect old runs,
- you want to share run review without retraining.

## 10) Export static dashboard snapshot

```bash
RUN_DIR=$(ls -1dt reports_quickstart/run_* | head -n 1)
python -m bnnr dashboard export \
  --run-dir "$RUN_DIR" \
  --out exported_dashboard
```

Open `exported_dashboard/index.html`.

## 11) One-shot mode (no live dashboard)

If you only need train + artifacts quickly:

```bash
python -m bnnr train \
  --config /tmp/bnnr_quickstart.yaml \
  --dataset cifar10 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --preset light \
  --without-dashboard
```

`--without-dashboard` disables only live server. Event logging remains enabled by CLI for post-run replay/export.

## 12) Loading trained model for inference

After training, load the best checkpoint for inference:

```python
import torch

ckpt = torch.load(
    "checkpoints/iter_1_augname.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

See [troubleshooting.md](troubleshooting.md) section 13 for details on checkpoint keys and PyTorch version notes.

## 13) Next pages

- `dashboard.md`
- `examples.md`
- `notebooks.md`
- `configuration.md`
- `cli.md`
- `api_reference.md`
- `troubleshooting.md`
