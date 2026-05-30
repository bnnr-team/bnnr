# Product roadmap

**Updated:** 2026-05-30 · **Current release:** v0.4.10

BNNR is a PyTorch vision toolkit: **train → explain → improve → prove** — with `bnnr analyze` (failure + XAI HTML report), ICD/AICD augmentations, and optional detection training via adapters.

---

## Shipped (baseline)

| Area | What exists today |
|------|-------------------|
| **Analyze** | `bnnr analyze` + portable HTML/JSON for classification and multilabel; [sample report](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html) |
| **Training** | CLI `train` / `demo` / `quickstart`; presets; live dashboard |
| **XAI aug** | ICD, AICD, branch search; [plugin_icd.md](plugin_icd.md) for custom loops |
| **Detection** | Train API + bbox augs; Ultralytics adapter; [detection.md](detection.md) — **no** `analyze` for detection yet |
| **Benchmarks** | CIFAR-10 (3 seeds): no aug vs RandAugment vs BNNR — [benchmarks.md](benchmarks.md) |
| **Integrations** | pytorch-grad-cam loop, Ultralytics quickstart — [integrations.md](integrations.md) |

---

## Q2 2026 (June–July) — analyze adoption

| Planned | Description |
|---------|-------------|
| **Torchvision → analyze** | Docs + runnable example: pretrained classifier checkpoint → `bnnr analyze` report (no full retrain) |
| **Getting started** | “Try analyze first” + githack sample links in [getting_started.md](getting_started.md) |
| **Upstream docs** | Official link in Ultralytics docs (after PR merge); grad-cam ecosystem follow-up |
| **Contributor templates** | GitHub issue forms: benchmark proposal, “Who uses BNNR” showcase |

---

## Q3 2026 — workflows & reach

| Planned | Description |
|---------|-------------|
| **Multilabel CLI** | `bnnr train` + YAML parity with multilabel API ([cli.md](cli.md)) |
| **Colab analyze** | One-click analyze-only notebook; README link |
| **HF / timm path** | Optional example: non–demo-CNN checkpoint + analyze (research persona) |
| **grad-cam tutorial** | Short upstream contribution (ICD loop), after maintainer response |
| **bnnr.dev** | Site CTA aligned with sample HTML + benchmark table |

---

## Q4 2026 — depth (if Q3 traction)

| Planned | Description |
|---------|-------------|
| **`bnnr compare`** | Side-by-side two checkpoints (pre/post fine-tune, compliance use case) |
| **More benchmarks** | Optional Fashion-MNIST / STL-10 rows in `benchmarks/results.json` |
| **Analyze HTML UX** | Interactive filters in report (only if users ask for it) |

---

## 2027 H1 — detection analyze (gated)

Start only when classification analyze + benchmarks are stable **and** there is clear demand (e.g. external issues).

| Planned | Description |
|---------|-------------|
| **`bnnr analyze` for detection** | Failure analysis on bbox predictions; docs + tests; explicit limits |
| **Ultralytics parity** | Examples and integration doc updates |

Until then: detection stays **train + ICD API** only; no marketing promise for detection analyze.

---

## Not planned (product)

| Item | Note |
|------|------|
| SaaS / multi-user experiment cloud | Local HTML/JSON audit only |
| YOLO trainer replacement | Adapter alongside Ultralytics |
| Label-noise platform (Cleanlab-scale) | Different problem; analyze + XAI focus |
| ImageNet SOTA claims | Benchmarks stay narrow with protocol caveats |

---

## How features land

Each user-facing addition should include: doc page (or section), example under `examples/` when runnable, and a README line if it is a primary entry point.

To contribute: [CONTRIBUTING.md](../CONTRIBUTING.md) · [open good first issues](https://github.com/bnnr-team/bnnr/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) · [Discussions](https://github.com/bnnr-team/bnnr/discussions)
