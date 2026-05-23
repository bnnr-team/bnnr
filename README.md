<p align="center">
  <img src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/logo.png" alt="BNNR Logo" width="180">
</p>

<p align="center">
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/v/bnnr?style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/pyversions/bnnr?style=flat-square" alt="Python"></a>
  <a href="https://github.com/bnnr-team/bnnr/stargazers"><img src="https://img.shields.io/github/stars/bnnr-team/bnnr?style=flat-square&logo=github" alt="GitHub stars"></a>
  <a href="https://pepy.tech/projects/bnnr"><img src="https://static.pepy.tech/personalized-badge/bnnr?period=total&amp;units=INTERNATIONAL_SYSTEM&amp;left_color=BLACK&amp;right_color=GREEN&amp;left_text=downloads" alt="PyPI downloads"></a>
  <a href="https://github.com/bnnr-team/bnnr/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bnnr-team/bnnr?style=flat-square" alt="License"></a>
  <a href="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml"><img src="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <video width="720" controls playsinline poster="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/hero-promo-poster.jpg">
    <source src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/hero-promo.mp4" type="video/mp4">
    <a href="https://www.bnnr.dev">Watch the demo on bnnr.dev</a>
  </video>
</p>
<p align="center"><em>Demo with audio (1280p). Original 4K recording: <a href="https://www.bnnr.dev">bnnr.dev</a></em></p>

# BNNR (Bulletproof Neural Network Recipe)

**Audit why your vision model fails — then fix it with XAI-guided augmentation.**

BNNR is an open-source PyTorch toolkit: structured reports, live dashboard, and saliency-driven augmentations (ICD / AICD) that test what actually improves validation metrics.

---

## Audit in 60 seconds (`bnnr analyze`)

**Lowest-friction entry:** run diagnostics on a checkpoint you already have — no retraining.

```bash
pip install bnnr
python3 -m bnnr analyze --model checkpoints/best.pt --data cifar10 --output ./analysis_out
# open ./analysis_out/report.html
```

Metrics, confusion patterns, XAI overlays, failure analysis, and recommendations. Details: [docs/analyze.md](docs/analyze.md).

---

## Same accuracy — different model behavior

Validation accuracy alone can hide shortcut learning. BNNR makes attention and branch decisions visible:

<p align="center">
  <img src="docs/assets/xai-same-accuracy-diff-behavior.png" alt="Same validation accuracy but different saliency and confidence after BNNR" width="800">
</p>
<p align="center"><em>Same val accuracy — different where the model looks (and confidence on the sample).</em></p>

<p align="center">
  <img src="docs/assets/xai-measurable-gains-wrong-to-correct.png" alt="Wrong to correct prediction with measurable branch selection" width="800">
</p>
<p align="center"><em>Only measurable improvements are kept along the selected path.</em></p>

---

## Train with a live loop

```bash
pip install "bnnr[dashboard]"
python3 -m bnnr demo   # ~1 min CIFAR-10 + dashboard at http://127.0.0.1:8080/
```

Supported tasks (**v0.4.4**): single-label classification, multi-label classification, object detection (COCO-mini / YOLO). See [Detection docs](docs/detection.md).

---

## XAI-driven augmentations (ICD & AICD)

BNNR uses saliency maps to guide augmentation — not random flips and crops.

<p align="center">
  <img src="docs/assets/icd-panel.png" alt="ICD — mask what the model looks at" width="720">
</p>
<p align="center"><strong>ICD</strong> — masks the regions the model already focuses on (highest saliency), forcing it to learn from context instead of shortcuts.</p>

<p align="center">
  <img src="docs/assets/aicd-panel.png" alt="AICD — mask what the model ignores" width="720">
</p>
<p align="center"><strong>AICD</strong> — masks low-saliency background and irrelevant textures, sharpening focus on discriminative features.</p>

---

## Benchmarks

| Dataset | Baseline | + BNNR | Gain |
|---------|----------|--------|------|
| CIFAR-10 | TBD | TBD | — |
| Fashion-MNIST | TBD | TBD | — |
| STL-10 | TBD | TBD | — |

*Run [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) to fill this table. See [benchmarks/README.md](benchmarks/README.md) and [docs/benchmarks.md](docs/benchmarks.md).*

---

## Quickstart

```bash
pip install "bnnr[dashboard]"

# Zero flags — CIFAR-10 demo CNN, ICD preset, live dashboard (~1 min)
python3 -m bnnr demo
```

Interactive wizard (prompts for dataset/preset; sample limits 128/64):

```bash
python3 -m bnnr quickstart
```

Full CLI training with built-in defaults:

```bash
python3 -m bnnr train --dataset cifar10 --preset light --with-dashboard
```

Open `http://127.0.0.1:8080/` for the live dashboard (QR code in terminal for mobile on the same Wi-Fi).

Advanced: pass `--config path.yaml` to override defaults.

---

## Live dashboard

Real metrics from a BNNR training run — branch tree, charts, XAI previews, and dataset insights.

| Overview | Branch Tree | Metrics |
|:---:|:---:|:---:|
| ![Dashboard Overview](docs/assets/dashboard-overview.png) | ![Branch tree — test improvements before you commit](docs/assets/branch-tree-improvements.png) | ![Metrics](docs/assets/dashboard-metrics.png) |

| Samples & XAI | Analysis | Dataset Insight |
|:---:|:---:|:---:|
| ![Samples and XAI](docs/assets/dashboard-samples.png) | ![Analysis](docs/assets/dashboard-analysis.png) | ![Dataset Insight](docs/assets/dashboard-insight.png) |

---

## What makes BNNR different

- **XAI-driven augmentation (ICD / AICD)** — augmentations guided by saliency maps; no other PyTorch toolkit combines explainability and data augmentation this way.
- **Auto-augmentation search** — iterative branching keeps only augmentations that measurably improve your validation metric.
- **Auditable reports** — structured JSON reports with metrics, XAI heatmaps, and branch decisions for stakeholders or compliance review.

---

## Links

| Resource | URL |
|----------|-----|
| Website | [bnnr.dev](https://bnnr.dev) |
| Documentation | [docs/README.md](docs/README.md) |
| Examples | [docs/examples.md](docs/examples.md) |
| Colab (classification) | [Open in Colab](https://colab.research.google.com/github/bnnr-team/bnnr/blob/main/examples/classification/bnnr_classification_demo.ipynb) |
| API reference | [docs/api_reference.md](docs/api_reference.md) |
| Model analysis (`bnnr analyze`) | [docs/analyze.md](docs/analyze.md) |

---

## Python API

```python
import bnnr

result = bnnr.quick_run(model, train_loader, val_loader)
print(result.best_metrics)
```

For a one-epoch smoke test: `bnnr.quick_run(..., m_epochs=1, max_iterations=1)`.

Advanced: [Golden path](docs/golden_path.md) (`BNNRTrainer`, custom adapters, detection). API details: [api_reference.md](docs/api_reference.md).

---

## Documentation

<details>
<summary><strong>Install from source, CLI reference, full doc index</strong></summary>

### Install from source

```bash
git clone https://github.com/bnnr-team/bnnr.git
cd bnnr
(cd dashboard_web && npm ci && npm run build)
pip install -e ".[dev,dashboard]"
```

The PyPI **wheel** ships the `bnnr` package only. Runnable scripts (`examples/`), notebooks, and the documentation tree (`docs/`) live in this repository.

### Main CLI commands

```bash
python3 -m bnnr --help
python3 -m bnnr demo
python3 -m bnnr train --help
python3 -m bnnr analyze --help
python3 -m bnnr report --help
python3 -m bnnr list-datasets
python3 -m bnnr list-augmentations -v
python3 -m bnnr list-presets
python3 -m bnnr dashboard serve --run-dir reports --port 8080
python3 -m bnnr dashboard export --run-dir reports/run_YYYYMMDD_HHMMSS --out exported_dashboard
```

### Doc index

- [Getting started](docs/getting_started.md)
- [Python quickstart (`quick_run`)](docs/quickstart_api.md)
- [Configuration](docs/configuration.md)
- [CLI](docs/cli.md)
- [Dashboard](docs/dashboard.md)
- [Augmentations](docs/augmentations.md)
- [Detection](docs/detection.md)
- [Analyze (standalone diagnostics)](docs/analyze.md)
- [Examples](docs/examples.md)
- [Notebooks](docs/notebooks.md)
- [Artifacts](docs/artifacts.md)
- [Troubleshooting](docs/troubleshooting.md)

### Requirements

- Python `>=3.10`
- Core: `torch`, `torchvision`, `numpy`, `typer`, `pydantic`, `pyyaml`, `grad-cam`
- Dashboard extra: `fastapi`, `uvicorn`, `websockets`, `qrcode`

</details>

---

## License

MIT License — use BNNR freely in research, production, and commercial projects.
