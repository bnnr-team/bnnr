<p align="center">
  <img src="docs/assets/logo.png" alt="BNNR Logo" width="180">
</p>

<p align="center">
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/v/bnnr?style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/pyversions/bnnr?style=flat-square" alt="Python"></a>
  <a href="https://github.com/bnnr-team/bnnr/stargazers"><img src="https://img.shields.io/github/stars/bnnr-team/bnnr?style=flat-square&logo=github" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/bnnr/"><img src="https://img.shields.io/pypi/dt/bnnr?label=downloads&style=flat-square" alt="PyPI downloads"></a>
  <a href="https://github.com/bnnr-team/bnnr/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bnnr-team/bnnr?style=flat-square" alt="License"></a>
  <a href="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml"><img src="https://github.com/bnnr-team/bnnr/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <video src="docs/assets/hero-promo.mp4" controls width="720" poster="docs/assets/hero-promo-poster.jpg">
  </video>
</p>

# BNNR (Bulletproof Neural Network Recipe)

**BNNR automatically improves your PyTorch vision models using XAI** — find what your model gets wrong, fix it with intelligent augmentation, and prove the result with structured reports and a live dashboard.

Supported tasks (**v0.3.0**): single-label classification, multi-label classification, and object detection (COCO-mini / YOLO). See [Detection docs](docs/detection.md).

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
| *Coming soon* | — | — | — |

Reproducible benchmark results on CIFAR-10, STL-10, and Fashion-MNIST will be published here. Track progress in [GitHub Issues](https://github.com/bnnr-team/bnnr/issues).

---

## Quickstart

```bash
pip install "bnnr[dashboard]"

python3 -m bnnr train --dataset cifar10 --preset light --with-dashboard
```

Interactive wizard (same built-in defaults, sample limits 128/64):

```bash
python3 -m bnnr quickstart
```

Open `http://127.0.0.1:8080/` for the live dashboard (QR code in terminal for mobile on the same Wi-Fi).

Advanced: pass `--config path.yaml` to override defaults.

---

## Live dashboard

Real metrics from a BNNR training run — branch tree, charts, XAI previews, and dataset insights.

| Overview | Branch Tree | Metrics |
|:---:|:---:|:---:|
| ![Dashboard Overview](docs/assets/dashboard-overview.png) | ![Branch Tree](docs/assets/dashboard-tree.png) | ![Metrics](docs/assets/dashboard-metrics.png) |

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

---

## Python API

```python
from bnnr import quick_run, BNNRConfig

result = quick_run(
    model,
    train_loader,
    val_loader,
    config=BNNRConfig(m_epochs=5, max_iterations=3, device="auto"),
)
print(result.best_metrics)
```

See [Golden path](docs/golden_path.md) and [API reference](docs/api_reference.md) for custom adapters and detection.

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
