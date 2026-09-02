# Benchmarks

Two reproducible augmentation comparisons:

1. **CIFAR-10 / demo CNN** (below) — fast, illustrative, runs on CPU.
2. **ResNet18 / Imagewoof** ([jump](#resnet18--imagewoof-benchmark)) — a fine-grained, low-data, from-scratch regime (where augmentation actually matters), with RandAugment **and** TrivialAugment baselines over 5 seeds. Cheap enough for a free Colab T4. Numbers land in the table after a GPU run.

---

## CIFAR-10 / demo CNN

Reproducible comparison of **three training setups** on the same demo CNN, dataset split, and epoch budget:

| Condition | What it is |
|-----------|------------|
| `no_bnnr` | Crop + flip only — no BNNR augmentations, no branch search |
| `randaugment` | **torchvision RandAugment** — random policy-based augmentations (external baseline) |
| `bnnr_branch_search` | Full **BNNR branch search** over **ICD**, **AICD**, and ChurchNoise |

## What we compare

1. **BNNR vs no augmentation** — does the branching system (with saliency-guided ICD/AICD) beat plain training?
2. **BNNR vs RandAugment** — does targeted, XAI-aware augmentation beat off-the-shelf random augs?
3. **Attention maps** — after each run, **OptiCAM** overlays on the **same 8 validation images** (`config.yaml` → `xai_val_indices`). Compare `runs/*/xai/attention_*.png` to see where each model looks.

Lower **edge ratio** and more focused **coverage** on the object usually indicate less background reliance.

## Layout

```
benchmarks/
  config.yaml      # shared epochs, metrics, RandAugment params, XAI indices
  lib.py           # conditions, training, attention export
  run.py           # CLI (resume-safe)
  summarize.py     # metrics + attention stats table
  results.json     # aggregated results (commit after review)
  runs/            # per-run logs + xai/ overlays (gitignored)
```

## Run

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Three seeds for publication-ready numbers:

```bash
python benchmarks/run.py --seeds 42,43,44 --device cpu
```

List conditions:

```bash
python benchmarks/run.py --list-conditions
```

## BNNR augmentations in this benchmark

| Name | What it masks | Right when |
|------|---------------|------------|
| **ICD** | The *most* salient tiles, above the threshold percentile | The model leans on a cue you want broken. ICD destroys what it currently relies on |
| **AICD** | The *least* salient tiles | Attention is already on the object and the surrounding context needs suppressing |
| **ChurchNoise** | Nothing saliency-guided; regional noise | Non-XAI candidate in the branch pool |

**The least salient tiles are the background only when the model already attends the object.** On a model attending the background, which is the case ICD and AICD exist to help with, AICD masks the object instead. The two are opposite operations on attention, so which one is appropriate depends on where attention already is.

BNNR does not currently diagnose that. It trains both candidates and keeps whichever scored higher on selection-validation accuracy, a criterion the T20 findings showed is close to orthogonal to the objective. Diagnosing the attention regime is bnnr-team/bnnr#403 and acting on it is bnnr-team/bnnr#413.

The branch search keeps augmentations that improve validation accuracy; the winning path is recorded in `results.json` → `best_path`.

## Shared fairness rules

- Same demo CNN (`_CifarCNN`), Adam lr=1e-3, batch 64, `m_epochs` from `config.yaml`
- Same random seed per condition within a run
- RandAugment uses `num_ops=2`, `magnitude=9` (torchvision defaults in `_benchmark`)
- Attention maps always use the same validation indices across conditions

**Protocol note:** `no_bnnr` and `randaugment` train for **5 epochs** only. `bnnr_branch_search` runs baseline (5 ep) plus candidate screening (up to 3×5 ep per iteration × 3 iterations) — **much more compute** and a different curriculum (augmentations added after baseline). Compare numbers as *“full BNNR product vs fixed-epoch baselines”*, not equal-budget ablation.

## Results (2026-05-28, seeds 42–44, CPU)

| Condition | Median val acc | Δ vs no BNNR | Per-seed |
|-----------|----------------|--------------|----------|
| Without BNNR (crop + flip) | **75.3%** | — | 75.3, 75.6, 75.3 |
| RandAugment (torchvision 2,9) | **72.5%** | −2.8 pp | 72.2, 72.5, 73.5 |
| BNNR branch search | **81.4%** | +6.1 pp | 81.4, 81.3, 81.6 |

Within-run BNNR gain vs its own baseline phase: +7.3 to +12.1 pp (`gain_vs_within_run_baseline_pp` in `results.json`). Winning paths varied by seed (e.g. ChurchNoise→ICD, or full ICD+AICD+ChurchNoise stack).

**Takeaways (honest):**

1. **BNNR full pipeline** clearly beats both baselines on this demo setup — stable ~81% across seeds.
2. **RandAugment at 5 epochs** underperforms crop+flip here; regularization likely needs longer training or lower `magnitude` — not a bug in integration.
3. **Attention (OptiCAM):** BNNR shows lower mean coverage (~13.5% vs ~18%); edge ratio mixed — qualitative XAI in `runs/*/xai/`, not a single headline metric.
4. **Do not claim SOTA or “beats RandAugment” without citing protocol** — demo CNN, short baselines, unequal epoch budget.

Raw data: [`results.json`](results.json). Regenerate table: `python benchmarks/summarize.py --markdown`.

### README figure

Side-by-side OptiCAM on the same val image (seed 44):

```bash
python scripts/build_benchmark_xai_readme_asset.py
```

Output: `docs/assets/benchmark-xai-comparison.png` (used in root README).

---

## ResNet18 / Imagewoof benchmark

A more convincing benchmark than the demo-CNN / CIFAR-10 table above: a **fine-grained** task (10 dog breeds from real ImageNet images) trained **from scratch** in a **low-data** regime — the setting where augmentation actually drives large, significant deltas, and where saliency-guided ICD/AICD have real spatial structure to act on (unlike 32px CIFAR).

| Condition | What it is |
|-----------|------------|
| `no_aug` | RandomResizedCrop + RandomHorizontalFlip — no extra augmentation |
| `randaugment` | + **torchvision RandAugment** (external baseline) |
| `trivialaugment` | + **torchvision TrivialAugmentWide** (parameter-free external baseline) |
| `bnnr_branch_search` | Full **BNNR branch search** over **ICD**, **AICD**, and ChurchNoise |

### Design

- **Dataset:** Imagewoof2-160 (fast.ai) — auto-downloaded. A **balanced 100 images/class** train subset, the **full val split as a fixed test set**. No cross-validation; **5 seeds** capture training variance.
- **Model:** ResNet18 from `torchvision.models`, **random init (from scratch)**. Imagewoof classes overlap ImageNet, so from-scratch is the honest default (`--pretrained` and `--arch resnet50` are available).
- **In-model normalization:** ImageNet mean/std are applied *inside* the model (registered buffers), so every condition feeds plain `ToTensor()` tensors in `[0, 1]` and BNNR's uint8-range ICD/AICD augmentations remain compatible.
- **Same** backbone, optimizer (SGD, momentum 0.9, weight decay 5e-4), cosine schedule, epochs, and seeds across all conditions — only the augmentation strategy varies.
- **OptiCAM** overlays on fixed Imagewoof val indices, exported per run (`runs_imagewoof/*/xai/`).

### Run in Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnnr-team/bnnr/blob/main/benchmarks/colab_imagewoof.ipynb)

Mount Google Drive, run all cells. Everything (metrics, XAI overlays, ZIP backup) lands on your Drive at `MyDrive/bnnr_benchmarks/`. Resume-safe — if the Colab session dies, re-run the full-benchmark cell and completed (condition, seed) pairs are skipped. The dataset is cached on Drive too, so it isn't re-downloaded after a restart.

ETA: **~1.5–2h on a free T4**, ~20 min on A100.

### Run locally

```bash
# Fast sanity check (CPU-friendly, tiny subset, img-size 64)
python benchmarks/run_imagewoof.py --smoke

# Full benchmark — 5 seeds, GPU (this is the publication run)
bash benchmarks/reproduce_imagewoof.sh
# or:
python benchmarks/run_imagewoof.py --seeds 42,43,44,45,46 --device cuda

# Write results + XAI overlays into a single directory (e.g. Drive / shared volume)
python benchmarks/run_imagewoof.py --seeds 42,43,44,45,46 --device cuda \
  --drive-base-dir /path/to/output

# Summarize
python benchmarks/summarize.py --results benchmarks/results_imagewoof.json --markdown
```

`run_imagewoof.py` checkpoints `results_imagewoof.json` after every (condition, seed) run, so the matrix is resume-safe — a crash or interruption keeps completed runs.

### Layout

```
benchmarks/
  run_imagewoof.py           # CLI (resume-safe matrix runner)
  reproduce_imagewoof.sh     # one-command full run (5 seeds)
  colab_imagewoof.ipynb      # one-click Colab (free T4)
  results_imagewoof.json     # aggregated results (commit after review)
  runs_imagewoof/            # per-run logs + xai/ overlays (gitignored)
```

### Protocol caveats (read before quoting numbers)

- **Unequal compute by design.** `bnnr_branch_search` runs a baseline phase plus branch search (more epochs of compute than the fixed-epoch baselines). Compare as *"full BNNR product vs fixed-epoch baselines"*, not equal-budget ablation.
- **Not an ImageNet-SOTA claim.** This is a low-data fine-grained transfer setup for comparing augmentation *strategies*, not a leaderboard entry.
- **Low-data, from-scratch by design.** A small balanced train subset trained from random init is what surfaces augmentation effects; `--train-per-class`, `--pretrained`, `--epochs`, and `--img-size` tune the regime.
- **Fill sweeps share baselines and need enough seeds.** Fill-independent conditions run once and are reused across every fill, so a sweep only multiplies the ICD/AICD-using conditions. Ranking many fills is Holm-corrected: with 5 fills the "meaningful" verdict needs **≥ 8 seeds**, and `summarize_grand.py` prints an explicit *underpowered* note when that gate can't fire.

### Results

_Pending a GPU run._ Run `reproduce_imagewoof.sh`, review `results_imagewoof.json`, then paste the `summarize.py --markdown` table here and into the root `README.md`. Do not hand-write numbers.

---

## Grand benchmark (paper-quality, equal-compute)

The Imagewoof benchmark above compares the **full BNNR product** against fixed-epoch baselines (unequal compute, single dataset). The grand benchmark is the stricter, publication-grade version designed to survive peer review: **equal total GPU-epoch budget across every condition**, a **held-out test split never used for model selection**, an explicit **XAI-ablation condition** (`bnnr_random`), and **cross-dataset generalization**.

### What it isolates

| Claim | Comparison |
|-------|-----------|
| XAI-guided selection beats random selection | `bnnr_xai` vs `bnnr_random` (the headline claim) |
| ICD / AICD each help on their own | `icd_only` / `aicd_only` vs `no_aug` |
| Branch search beats the best single augmentation | `bnnr_xai` vs best `*_only` |
| BNNR beats strong external baselines | vs RandAugment, TrivialAugment, AutoAugment |
| ICD reduces shortcut learning | XAI metrics: edge ratio, gini, coverage |
| Results generalize across domains | 6 datasets |
| Which ICD/AICD fill strategy is best | `icd_only` across `--fill-strategies` |

### Conditions (10)

`no_aug`, `randaugment`, `trivialaugment`, `autoaugment`, `churchnoise_only`, `icd_only`, `aicd_only`, `icd_aicd_fixed`, `bnnr_random`, `bnnr_xai`.

### Datasets

| Dataset | Classes | Train/class | Res | Domain |
|---------|---------|-------------|-----|--------|
| Imagewoof | 10 | 100 | 128px | Fine-grained animals |
| Oxford Pets | 37 | 100 | 224px | Fine-grained animals |
| Flowers102 | 102 | 10 | 224px | Flowers (extreme low-data) |
| DTD | 47 | 120 | 224px | Textures |
| FGVC-Aircraft | 100 | 33 | 224px | Aircraft |
| EuroSAT | 10 | 100 | 64px | Satellite |

### Fill strategy

| Value | Fill |
|-------|------|
| `gaussian_blur` | Blurred copy of the region (**default**) |
| `local_mean` | Per-region mean color |
| `global_mean` | Whole-image mean color |
| `noise` | Random noise (seed-controlled) |
| `solid` | Constant fill |

### Protocol caveats (read before quoting numbers)

- **Equal total compute, not equal per-model epochs.** Every condition spends the same total GPU-epoch budget `B`. The `bnnr_xai` / `bnnr_random` *final* model is trained for 2 phases (baseline + chosen candidate), while single-aug and baseline conditions pour all `B` epochs into one model. Compute is matched; the per-model epoch count is not. This is **conservative** for the "branch search > best single aug" claim.
- **Held-out test is reserved for final reporting only.** Every condition selects its best epoch on `selection_val`; the held-out test split is evaluated exactly once. No condition early-stops on the test set.
- **`bnnr_random` is the XAI ablation.** Identical compute and augmentation pool as `bnnr_xai`, but the candidate is chosen at random (seed-controlled). `bnnr_xai` vs `bnnr_random` isolates the contribution of XAI-guided selection from the augmentation pool itself.
- **From-scratch, low-data by design:** the regime where augmentation strategy actually moves the needle, not a leaderboard entry.

### Run

```bash
# Sanity check (CPU, tiny subset): exercises every code path
python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke

# Primary run: Imagewoof, 10 conditions x 10 seeds (GPU)
python benchmarks/run_grand_benchmark.py --dataset imagewoof --device cuda

# Generalization datasets (6 conditions x 7 seeds each)
python benchmarks/run_grand_benchmark.py --dataset pets --device cuda

# Summarize: Wilcoxon signed-rank + bootstrap CI + Holm-Bonferroni
python benchmarks/summarize_grand.py --results-dir benchmarks/ --markdown
# Same as above but "does any fill beat gaussian_blur?"
python benchmarks/summarize_grand.py --results-dir benchmarks/ --markdown --reference-fill gaussian_blur

# Fill-strategy ablation: sweep all five fills for icd_only
python benchmarks/run_grand_benchmark.py --dataset imagewoof --device cuda --conditions icd_only --fill-strategies gaussian_blur,local_mean,global_mean,noise,solid
```

Resume-safe: `results_{dataset}_{regime}.json` is checkpointed after every `(condition, seed)` run. Use `--drive-base-dir` to land results, run logs, and dataset cache under one directory (Colab/Drive).

### Layout

```
benchmarks/
  run_grand_benchmark.py   # CLI (resume-safe, equal-compute, held-out test)
  dataset_loaders.py       # 6-dataset registry, 3-way train/selection_val/held_out_test split
  metrics_extended.py      # F1-macro, Top-5, Cohen's kappa, ECE, XAI metrics
  summarize_grand.py       # Wilcoxon + bootstrap CI + Holm-Bonferroni table
  runs_grand/              # per-run logs + xai/ overlays (gitignored)
```

### Results

Imagewoof, from scratch, equal compute (budget 40), 10 seeds, 100 records in
[`results_imagewoof_scratch.json`](results_imagewoof_scratch.json). Regenerate with:

```bash
python benchmarks/summarize_grand.py --results-dir benchmarks/ --datasets imagewoof --markdown
```

| Condition | Median | ±IQR | mean±std | n | Δ vs no_aug | p (Holm) vs bnnr_xai | r | Bootstrap 95% CI | ECE ↓ | GPU-epochs |
|-----------|--------|------|----------|---|------------|---------------------|---|-----------------|-------|-----------|
| No augmentation (crop + flip) | 33.97% | ±1.38pp | 34.25% ±1.20 | 10 | — | p=0.018 * (exact) | -1.00 | [-5.90, -4.53]pp | 0.257 | 40 |
| RandAugment (torchvision) | 30.79% | ±2.11pp | 31.18% ±1.30 | 10 | -3.17pp | p=0.018 * (exact) | -0.96 | [-3.77, -0.66]pp | 0.033 | 40 |
| TrivialAugmentWide (torchvision) | 31.67% | ±1.76pp | 31.74% ±1.58 | 10 | -2.30pp | p=0.041 * (exact) | -0.85 | [-3.91, -0.71]pp | 0.042 | 40 |
| AutoAugment (ImageNet policy) | 27.83% | ±5.07pp | 27.09% ±3.16 | 10 | -6.13pp | p=0.645 ns (exact) | 0.38 | [-0.88, +6.18]pp | 0.025 | 40 |
| ChurchNoise only (non-XAI ablation) | 33.40% | ±2.70pp | 33.94% ±1.97 | 10 | -0.57pp | p=0.018 * (exact) | -1.00 | [-6.67, -3.13]pp | 0.246 | 40 |
| ICD only | 33.83% | ±2.35pp | 33.80% ±1.73 | 10 | -0.14pp | p=0.018 * (exact) | -1.00 | [-6.58, -3.32]pp | 0.240 | 40 |
| AICD only | 33.14% | ±1.35pp | 33.29% ±1.36 | 10 | -0.83pp | p=0.018 * (exact) | -1.00 | [-5.53, -3.27]pp | 0.250 | 40 |
| ICD+AICD fixed (no search) | 33.24% | ±2.00pp | 33.05% ±1.48 | 10 | -0.72pp | p=0.018 * (exact) | -1.00 | [-5.22, -3.39]pp | 0.230 | 40 |
| BNNR random selection (XAI ablation) | 29.62% | ±1.79pp | 28.97% ±2.02 | 10 | -4.35pp | p=0.945 ns (exact, n=8) | -0.06 | [-0.20, +0.15]pp | 0.054 | 40 |
| BNNR XAI-guided (equal compute) | 29.51% | ±1.56pp | 28.95% ±1.92 | 10 | -4.46pp | — | — | — | 0.060 | 40 |

**Headline:** `bnnr_xai` (29.51%) vs `bnnr_random` (29.62%) — median of paired
differences **+0.00pp**, n=10 (8 after dropping two tied pairs), p=0.945 ns,
r=−0.06, bootstrap 95% CI [−0.20, +0.15]pp. **XAI-guided candidate selection is
indistinguishable from random selection at n=10.** Full write-up:
[`findings_t20/findings_imagewoof.md`](findings_t20/findings_imagewoof.md).

Read the two Δ conventions apart: `Δ vs no_aug` compares a condition against
`no_aug` and is a difference of condition medians, descriptive and untested; the
`p`, `r` and CI in the same row compare `bnnr_xai` against that condition and the
CI is the median of the paired differences. The summarizer prints this
distinction in its own statistical notes. Estimators: [`stats.py`](stats.py).

Regenerated after #398, which fixed the rank-biserial sign and made the Δ and CI
paired. Do not hand-write numbers here.
