<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/logo-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/logo-light.png" />
    <img alt="BNNR Logo" src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/logo-dark.png" width="180" />
  </picture>
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
  <a href="https://www.bnnr.dev/">Website</a>
  ·
  <a href="https://www.linkedin.com/company/bnnr/">LinkedIn</a>
</p>

<p align="center">
  <a href="https://www.bnnr.dev">
    <img src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/hero-promo-poster.jpg" alt="Watch BNNR demo with audio on bnnr.dev" width="720">
  </a>
</p>
<p align="center"><em>Full demo with audio (4K): <a href="https://www.bnnr.dev">bnnr.dev</a></em></p>

# BNNR (Bulletproof Neural Network Recipe)

<p align="center"><strong>Train → Explain → Improve → Prove</strong></p>

**BNNR automatically improves your PyTorch vision models using XAI** — find what your model gets wrong, fix it with intelligent augmentation, and prove the result with structured reports and a live dashboard.

Supported tasks (**v0.4.11**): single-label classification, multi-label classification, and object detection (COCO-mini / YOLO). See [Documentation](https://github.com/bnnr-team/bnnr/blob/main/docs/README.md).

**Sample analyze report (no install):** [live HTML preview](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html)

---

## Quickstart

```bash
pip install "bnnr[dashboard]"

# Zero flags — CIFAR-10 demo CNN, ICD preset, live dashboard (~1 min)
python3 -m bnnr demo
```

```bash
python3 -m bnnr quickstart
```

```bash
python3 -m bnnr train --dataset cifar10 --preset light --with-dashboard
```

Open `http://127.0.0.1:8080/` for the live dashboard.

**Already have a checkpoint?** `python3 -m bnnr analyze --model checkpoints/best.pt --data cifar10 --output ./analysis_out` — [docs](https://github.com/bnnr-team/bnnr/blob/main/docs/analyze.md).

---

## XAI-driven augmentations (ICD & AICD)

BNNR uses saliency maps to guide augmentation — not random flips and crops.

<p align="center">
  <img src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/icd-panel.png" alt="ICD — mask what the model looks at" width="720">
</p>
<p align="center"><strong>ICD</strong> — masks the regions the model already focuses on (highest saliency), forcing it to learn from context instead of shortcuts.</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/aicd-panel.png" alt="AICD — mask what the model ignores" width="720">
</p>
<p align="center"><strong>AICD</strong> — masks low-saliency background and irrelevant textures, sharpening focus on discriminative features.</p>

---

## Benchmarks

| Dataset | Without BNNR | BNNR branch search | RandAugment |
|---------|--------------|--------------------|-------------|
| CIFAR-10 | 75.3% | 81.4% | 72.5% |

Median val accuracy, 3 seeds, demo CNN ([methodology](https://github.com/bnnr-team/bnnr/blob/main/benchmarks/README.md)).

<p align="center">
  <img src="https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/benchmark-xai-comparison.png" alt="CIFAR-10 val 127: original input and OptiCAM — no BNNR vs RandAugment vs BNNR" width="720">
</p>

---

## Live dashboard

Real metrics from a BNNR training run — branch tree, charts, XAI previews, and dataset insights.

| Overview | Branch Tree | Metrics |
|:---:|:---:|:---:|
| ![Dashboard Overview](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-overview.png) | ![Branch Tree](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-tree.png) | ![Metrics](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-metrics.png) |

| Samples & XAI | Analysis | Dataset Insight |
|:---:|:---:|:---:|
| ![Samples and XAI](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-samples.png) | ![Analysis](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-analysis.png) | ![Dataset Insight](https://raw.githubusercontent.com/bnnr-team/bnnr/main/docs/assets/dashboard-insight.png) |

---

## What makes BNNR different

- **XAI-driven augmentation (ICD / AICD)** — augmentations guided by saliency maps; no other PyTorch toolkit combines explainability and data augmentation this way.
- **Auto-augmentation search** — iterative branching keeps only augmentations that measurably improve your validation metric.
- **Auditable reports** — structured JSON reports with metrics, XAI heatmaps, and branch decisions for stakeholders or compliance review.

---

## Links

| Resource | URL |
|----------|-----|
| Website | [bnnr.dev](https://www.bnnr.dev/) |
| LinkedIn | [BNNR on LinkedIn](https://www.linkedin.com/company/bnnr/) |
| Documentation | [docs/README.md](https://github.com/bnnr-team/bnnr/blob/main/docs/README.md) |
| Examples | [docs/examples.md](https://github.com/bnnr-team/bnnr/blob/main/docs/examples.md) |
| Roadmap (Q2–Q3 2026) | [docs/roadmap.md](https://github.com/bnnr-team/bnnr/blob/main/docs/roadmap.md) |
| Colab (classification) | [Open in Colab](https://colab.research.google.com/github/bnnr-team/bnnr/blob/main/examples/classification/bnnr_classification_demo.ipynb) |
| API reference | [docs/api_reference.md](https://github.com/bnnr-team/bnnr/blob/main/docs/api_reference.md) |
| Model analysis (`bnnr analyze`) | [docs/analyze.md](https://github.com/bnnr-team/bnnr/blob/main/docs/analyze.md) |
| Integrations (Grad-CAM, Ultralytics YOLO) | [docs/integrations.md](https://github.com/bnnr-team/bnnr/blob/main/docs/integrations.md) |
| Sample analyze report (live HTML) | [raw.githack.com preview](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html) |
| GitHub Discussions | [Q&A and showcase](https://github.com/bnnr-team/bnnr/discussions) |

---

## Python API

```python
import bnnr

result = bnnr.quick_run(model, train_loader, val_loader)
print(result.best_metrics)
```

Advanced: [Golden path](https://github.com/bnnr-team/bnnr/blob/main/docs/golden_path.md) and [API reference](https://github.com/bnnr-team/bnnr/blob/main/docs/api_reference.md).

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

- [Getting started](https://github.com/bnnr-team/bnnr/blob/main/docs/getting_started.md)
- [Configuration](https://github.com/bnnr-team/bnnr/blob/main/docs/configuration.md)
- [CLI](https://github.com/bnnr-team/bnnr/blob/main/docs/cli.md)
- [Dashboard](https://github.com/bnnr-team/bnnr/blob/main/docs/dashboard.md)
- [Augmentations](https://github.com/bnnr-team/bnnr/blob/main/docs/augmentations.md)
- [Detection](https://github.com/bnnr-team/bnnr/blob/main/docs/detection.md)
- [Analyze (standalone diagnostics)](https://github.com/bnnr-team/bnnr/blob/main/docs/analyze.md)
- [Examples](https://github.com/bnnr-team/bnnr/blob/main/docs/examples.md)
- [Notebooks](https://github.com/bnnr-team/bnnr/blob/main/docs/notebooks.md)
- [Artifacts](https://github.com/bnnr-team/bnnr/blob/main/docs/artifacts.md)
- [Troubleshooting](https://github.com/bnnr-team/bnnr/blob/main/docs/troubleshooting.md)

### Requirements

- Python `>=3.10`
- Core: `torch`, `torchvision`, `numpy`, `typer`, `pydantic`, `pyyaml`, `grad-cam`
- Dashboard extra: `fastapi`, `uvicorn`, `websockets`, `qrcode`

</details>

---

## Citation

If you use BNNR in academic work, cite the repository: [docs/citation.md](https://github.com/bnnr-team/bnnr/blob/main/docs/citation.md) (BibTeX). GitHub: **Cite this repository** via `CITATION.cff`.

---

## License

MIT License — use BNNR freely in research, production, and commercial projects.
