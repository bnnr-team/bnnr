# Troubleshooting

## What you will find here
Real failure modes observed in current BNNR runs, with reproducible checks and fixes.

## When to use this page
Use this when a command fails, a run hangs, dashboard is empty, or export output is unexpected.

## 1) Training finishes but process does not exit

Cause:

- run started with `--with-dashboard` (default), so live server stays active.

Fix:

- stop with `Ctrl+C`, or
- run one-shot with `--without-dashboard`.

## 2) `Dashboard backend dependencies are missing`

Cause:

- dashboard extras not installed.

Fix:

```bash
python3 -m pip install -e ".[dashboard]"
```

## 3) `--data-path is required ...`

Cause:

- missing path for dataset types that require external structure.

Applies to:

- `imagefolder`

Fix: pass `--data-path`.

## 4) Dashboard shows zero runs

Cause:

- wrong `--run-dir`, or run directory missing `events.jsonl`.

Fix:

- use a parent folder containing `run_*` directories, or
- point directly at a run directory that has `events.jsonl`.

## 6) (Historical) Python 3.9 and union types in CLI/FastAPI

**Supported releases (0.1.1+)** require **Python >=3.10**; this scenario does not apply to current wheels.

If you run an **older checkout** on Python 3.9, you could see `unsupported operand type(s) for |` from runtime-evaluated `X | None` annotations. Mitigations were: compatible annotations in introspected paths, or `eval-type-backport` for `python < 3.10` (removed when 3.9 support was dropped).

## 7) CI/test import error: `ModuleNotFoundError: No module named 'httpx'`

Cause:

- async dashboard tests require `httpx`.

Fix:

- include `httpx` in test/dev dependencies.

## 8) `pip install -e ".[dashboard]"` or `python -m build` fails in restricted environments

Cause:

- isolated build environment cannot fetch build backend packages (for example `hatchling`) due network restrictions.

Fix:

- in restricted/offline environments, use prepared build env and `python -m build --no-isolation`,
- in GitHub CI (network-enabled), standard `python -m build` should work.

## 9) CUDA appears unavailable

Check:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Fix:

- install CUDA-compatible PyTorch build for your OS/driver.

## 10) Augmentation instability on grayscale datasets

Symptom:

- shape/broadcast errors with aggressive presets on grayscale data.

Fix:

- start with RGB datasets for preset stress tests,
- keep grayscale smoke runs minimal and conservative.

## 11) `python3 -m venv` fails with `ensurepip is not available`

Cause:

- Python venv support package is missing on the host OS.

Fix:

- install system venv package (for example `python3.12-venv` on Ubuntu),
- recreate the virtual environment and continue quickstart steps.

## 12) QR code is visible but dashboard does not open on phone

Cause:

- phone is on a different network,
- local firewall blocks dashboard port,
- router/client isolation blocks peer-to-peer traffic.

Fix:

1. confirm phone and machine are on the same Wi-Fi,
2. open Network URL manually on phone (not only QR scan),
3. test with explicit port, e.g. `--dashboard-port 8080`,
4. allow Python/server traffic for that port in firewall settings.

## 13) Loading BNNR checkpoints for inference

BNNR checkpoints include RNG state for deterministic resume. When loading
for inference with PyTorch >= 2.6, pass `weights_only=False`:

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

Checkpoint keys: `model_state`, `iteration`, `augmentation_name`, `metrics`,
`config_snapshot`, `rng_state` (safe to ignore for inference).

## 14) Choosing XAI target layers for SimpleTorchAdapter

By default, `SimpleTorchAdapter` picks the last `Conv2d` layer for XAI.
To override, pass `target_layers` explicitly:

```python
# Example: EfficientNet-B0
target_layer = model.features[-1][0]  # last MBConv block

adapter = SimpleTorchAdapter(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    target_layers=[target_layer],
)
```

Common choices:

- ResNet: `model.layer4[-1]`
- EfficientNet: `model.features[-1][0]`
- ViT: last attention block (may require custom wrapper)

## 15) Windows: `RuntimeError: An attempt has been made to start a new process`

Cause:

- PyTorch `DataLoader` with `num_workers > 0` on Windows requires `if __name__ == "__main__":` guard.

Fix:

- wrap your training script entry point:

```python
if __name__ == "__main__":
    main()
```

- or set `num_workers=0` in your DataLoader.

## 16) Multi-label: `task: multilabel` in YAML but training still looks single-label

Cause:

- `bnnr train` preset pipelines (`build_*_pipeline` in `pipelines.py`) always use `CrossEntropyLoss` and single-label targets, regardless of `task` in the config file.

Fix:

- integrate multi-label data with `SimpleTorchAdapter(multilabel=True)` and `BCEWithLogitsLoss` ([golden_path.md](golden_path.md)), or run `examples/multilabel/multilabel_demo.py`; do not expect `--dataset cifar10` (or similar) alone to become multi-label.

## 17) Detection XAI with YOLO / Ultralytics

Symptom:

- Log says detection XAI or probe prediction snapshots were skipped for an Ultralytics backbone.

Cause:

- BNNR needs the high-level `YOLO.predict` path and the same image scaling as training. A raw `ultralytics.nn.tasks.DetectionModel` passed through `DetectionAdapter` does not expose that.

Fix:

- Use `UltralyticsDetectionAdapter` from `bnnr.detection_adapter` for Ultralytics YOLO training. That adapter implements `predict_detection_dicts` so detection XAI (activation / occlusion) and probe snapshots work with `xai_enabled=True`.
