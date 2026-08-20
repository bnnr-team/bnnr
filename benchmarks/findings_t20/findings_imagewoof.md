# T20 Findings — Imagewoof Grand Benchmark

**Task:** T20 — run the primary Imagewoof matrix (the headline evidence).
**Run:** `benchmarks/run_grand_benchmark.py --dataset imagewoof --device cuda`
**Regime:** scratch (from-scratch ResNet18, `img_size=128`, `train_per_class=100`, `budget=40`)
**Seeds:** 42–51 (n=10 per condition)
**Records:** 100/100 valid, zero errors
**Raw data:** `benchmarks/results_imagewoof_scratch.json`

This document covers the Imagewoof study. Its companion, `findings_waterbirds.md`, covers the repair regime on Waterbirds, and `findings.md` synthesises both.

---

## Headline result: bnnr_xai vs bnnr_random

| | bnnr_xai | bnnr_random | Δ | p (Holm) | r | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| Median held-out accuracy | 29.51% | 29.62% | −0.11pp | 0.945 (ns) | −0.06 | [−0.37, +0.27]pp |

**No detectable difference.** The 95% CI is under a percentage point wide and straddles zero. Per-seed, `bnnr_xai` beat `bnnr_random` on 4 of 10 seeds, lost on 4, tied on 2 — as close to a coin flip as ten seeds can show. This is the comparison the benchmark exists to make: it isolates XAI-guided candidate selection from everything else, since `bnnr_xai` and `bnnr_random` share identical compute, an identical candidate pool, and differ only in how the winning candidate is chosen. At n=10 we cannot distinguish XAI-guided selection from picking a candidate at random.

*(Verified directly against `results_imagewoof_scratch.json`: all 100 records present, zero errors, identical hardware — RTX 5080 Laptop GPU, PyTorch 2.12.1+cu130 — across every run, and an independent recomputation of the medians from raw `held_out_test_metric` values matches the `summarize_grand.py` output exactly.)*

> **Note on the effect size.** `summarize_grand.py` prints `r = 0.53` for this contrast. That value is wrong and the r column reported here is a corrected recomputation. The tool's own statistical note gives its formula as `1 - 2W/(n(n+1))`; the matched-pairs rank-biserial is `(W⁺ − W⁻)/(W⁺ + W⁻)`, which for `W = min(W⁺, W⁻)` is `1 - 4W/(n(n+1))`. The missing factor of two means the printed statistic is the *proportion* of positive ranks rather than a correlation, and its range is [0.5, 1.0] — it cannot express a null. Recomputed from the raw per-seed accuracies: W⁺ = 17, W⁻ = 19, so **r = −0.06**, which is what p = 0.945 requires. The conversion for any previously published table from this tool is `true r = 2 × printed r − 1`; see the table below and the separate issue.

### What "XAI-guided selection" means here, precisely

`selection_mode="xai"` does not use saliency maps to choose between candidates directly. It trains all three candidates (ICD, AICD, ChurchNoise) for `epochs_per_phase` (10) epochs from a shared baseline checkpoint, then greedily keeps whichever scored highest on `selection_val` accuracy. `selection_mode="random"` picks one of the same three candidates irrespective of score. The XAI/saliency machinery lives *inside* ICD and AICD — it conditions *where in the image* augmentation is applied — not in the arbitration between candidates.

Two things make a near-tie plausible rather than surprising:

- The three candidates are close in standalone quality (`icd_only` 33.83%, `aicd_only` 33.14%, `churchnoise_only` 33.40% — a 0.7pp spread). Greedy best-of-3 has little to exploit if the candidates are this close.
- Across all 100 runs, the saliency-derived `edge_ratio` diagnostic (fraction of saliency mass falling in the outer 15% border of the image versus the centre) shows essentially no correlation with held-out accuracy (Spearman ρ = 0.017). Whatever the saliency maps are doing at this stage of training, it is not tracking model quality in a way this benchmark can pick up.

## Broader picture: bnnr_xai/bnnr_random vs. every other condition

| Condition | Median | Mean ± SD | n | Δ vs no_aug | p (Holm) vs bnnr_xai | r (corrected) | 95% CI (bnnr_xai − condition) |
|---|---|---|---|---|---|---|---|
| No augmentation (crop + flip) | 33.97% | 34.25 ± 1.20 | 10 | — | 0.018 * | 1.00 | [−5.60, −3.84]pp |
| ICD only | 33.83% | 33.80 ± 1.73 | 10 | −0.14pp | 0.018 * | 1.00 | [−5.71, −3.50]pp |
| ChurchNoise only (non-XAI ablation) | 33.40% | 33.94 ± 1.97 | 10 | −0.57pp | 0.018 * | 1.00 | [−6.19, −2.89]pp |
| ICD+AICD fixed (no search) | 33.24% | 33.05 ± 1.48 | 10 | −0.72pp | 0.018 * | 1.00 | [−5.19, −2.61]pp |
| AICD only | 33.14% | 33.29 ± 1.36 | 10 | −0.83pp | 0.018 * | 1.00 | [−5.11, −2.69]pp |
| TrivialAugmentWide (torchvision) | 31.67% | 31.74 ± 1.58 | 10 | −2.30pp | 0.041 * | 0.86 | [−4.02, −0.82]pp |
| RandAugment (torchvision) | 30.79% | 31.18 ± 1.30 | 10 | −3.17pp | 0.018 * | 0.96 | [−3.33, −0.40]pp |
| BNNR random selection (XAI ablation) | 29.62% | 28.97 ± 2.02 | 10 | −4.35pp | 0.945 ns | −0.06 | [−0.37, +0.27]pp |
| **BNNR XAI-guided (equal compute)** | **29.51%** | 28.95 ± 1.92 | 10 | −4.46pp | — | — | — |
| AutoAugment (ImageNet policy) | 27.83% | 27.09 ± 3.16 | 10 | −6.13pp | 0.645 ns | 0.38 | [−0.71, +6.14]pp |

All conditions consumed a median of 40 GPU-epochs — the equal-compute constraint held.

The r column is corrected per the note above; the values printed by `summarize_grand.py` for these rows were 1.00, 1.00, 1.00, 1.00, 1.00, 0.93, 0.98, 0.53, —, 0.69. Rows where every seed-pair points the same way are unaffected by the bug (both formulas give 1.00 when W = 0); the two rows it distorts most are exactly the two weak ones, where a null-ish result was printed as a moderate effect.

Two structural readings. First, **no augmentation wins outright** in this regime — every augmentation condition, BNNR's included, sits at or below plain crop-and-flip. Second, the ICD-family conditions reach r = 1.00 against `bnnr_xai` despite clustering within 0.7pp of each other: every one of the ten seed-pairs pointed the same way, which is enough for Wilcoxon even when the practical gap is small. The gap that matters is the ~4–5pp separating the whole non-BNNR cluster from the BNNR cluster, and the section below explains it.

## Calibration tells a different story than accuracy

`test_ece` (expected calibration error, lower is better) is recorded per run but absent from the summary tables:

| Condition family | Median accuracy | Median ECE |
|---|---|---|
| torchvision policies (RandAugment / TrivialAugment / AutoAugment) | 27.8–31.7% | **0.024–0.042** |
| BNNR (`bnnr_xai` / `bnnr_random`) | 29.5% | 0.054–0.060 |
| no_aug, ICD-only, AICD-only, ChurchNoise-only, ICD+AICD fixed | 33.1–34.0% | **0.230–0.257** |

The most accurate conditions are the worst calibrated by roughly an order of magnitude. A plausible reading — that broad, high-diversity per-sample policies regularise predicted confidence even where they do not help top-line accuracy, while narrow transforms and no augmentation let the model become accurate but overconfident — is a hypothesis, not something isolated here. The defensible conclusion is narrower and still important: **accuracy alone understates how much these conditions differ, and the ranking is not robust to the choice of metric.**

## Why BNNR conditions underperform: an accounting artifact, not (necessarily) a bad method

The run's own metadata states it (`protocol_caveats[4]`): under equal total compute the BNNR final model trains for two phases — baseline (10 epochs) plus winning candidate (10 epochs) = 20 epochs — while single-augmentation and baseline conditions train the full 40 on one model. `total_gpu_epochs = 40` is identical everywhere, but that counts epochs *spent*, not epochs that end up in the deployed model. The other 20 bought comparison information.

The asymmetry compounds: plain conditions report their best epoch on `selection_val` out of up to 40 within-run checkpoints (median best epoch 35.5), while BNNR picks from three end-of-phase candidates.

## Follow-up: epoch-matched re-run (budget=80, final model = 40 epochs)

To test whether the epoch-split explains the gap rather than the search/selection process itself being harmful, `bnnr_xai` and `bnnr_random` were re-run on the same 10 seeds with `--budget 80` (`epochs_per_phase=20`, so final model = baseline(20) + winning candidate(20) = 40 epochs, matching every other condition). Note this is not a single-variable isolation: it also matures the baseline model before the XAI cache is computed (20 epochs instead of 10), so final-model epoch count and baseline maturity change together.

| | bnnr_xai | bnnr_random | Δ | p (Holm) | Bootstrap 95% CI |
|---|---|---|---|---|---|
| budget=40, original (final model = 20 epochs) | 29.51% | 29.62% | −0.11pp | 0.945 (ns) | [−0.37, +0.27]pp |
| budget=80, epoch-matched (final model = 40 epochs) | 33.88% | 33.50% | +0.39pp | 0.461 (ns) | [−0.63, +1.87]pp |

*(The r values printed for these two rows, 0.53 and 0.67, are subject to the formula defect described above; the corrected values are −0.06 and +0.34. Neither changes the conclusion, since both p-values are far from significance.)*

**The gap vs. `no_aug` (33.97%) nearly closes:** 4.46pp → 0.09pp for `bnnr_xai`, 4.35pp → 0.47pp for `bnnr_random`. This is a clean confirmation that the epoch-split, not the search-and-select mechanism itself, was the dominant driver of BNNR's apparent underperformance against the plain conditions in the main matrix.

`bnnr_xai` vs `bnnr_random` is still not statistically significant (p = 0.461, n=10) — matching final-model epochs did not produce a detectable XAI-guidance effect (5 wins / 3 losses / 2 ties, up from 4/4/2 at budget=40, still close to chance). The point estimate did flip sign (+0.39pp vs −0.11pp) and the bootstrap CI shifted from centred on zero to skewed positive ([−0.63, +1.87]pp), but this is one re-run at the same sample size, not a replication with more seeds. Read it as *not inconsistent with* a small positive effect, not as evidence of one — the calibrated conclusion is still "no detectable difference at n=10".

One secondary signal worth flagging without over-reading it: which candidate actually gets picked. At budget=40, `bnnr_xai`'s choices were spread across all three candidates (ChurchNoise 4, AICD 4, ICD 2) — about as varied as `bnnr_random`'s (ChurchNoise 2, ICD 4, AICD 4), consistent with the candidates being close in quality. At budget=80, `bnnr_xai` shifts toward ChurchNoise (6) and ICD (4) and **never** picks AICD, while `bnnr_random` stays roughly uniform. A more mature baseline seems to make the greedy selection more decisive — though this shows up as a change in *which* candidate wins, not yet in *whether* xai beats random.

The `edge_ratio`/accuracy correlation on this 20-run subset (ρ = −0.074) is consistent with the near-zero correlation on the full 100-run dataset (ρ = 0.017) — same conclusion, smaller sample.

Raw data kept deliberately outside `benchmarks/`: `~/bnnr_equal_epochs/results_imagewoof_scratch.json`. A file matching `results_*.json` inside `benchmarks/` would be swept into any future `summarize_grand.py --results-dir benchmarks/` call and would silently mix budget=40 and budget=80 runs under the same condition label.

## What ICD and AICD actually do — and a documentation defect

Verified by reading `src/bnnr/icd.py`:

```
ICD  — masks the MOST salient tiles   (invert_mask = False; threshold at the 75th percentile)
AICD — masks the LEAST salient tiles  (invert_mask = True;  threshold at the 25th percentile)
```

ICD destroys what the model currently relies on. AICD destroys what the model currently ignores. These are opposite operations on the model's own attention, not two flavours of one idea.

`benchmarks/README.md` describes AICD as masking "low-saliency **background** — reduces shortcut learning on context". The first half is right; the word *background* is an assumption rather than a description. Low-saliency equals background **only when the model already attends the object**. On a model that leans on the background — the shortcut-affected model the method is advertised to repair — the low-saliency region contains the object, and AICD masks the object while preserving the shortcut cue.

This matters beyond wording: it is the assumption a user forms when choosing an augmentation, and it is inverted precisely in the use case the tool is sold for. Recommended fix: describe both augmentations by their operation, and state the condition under which each is the appropriate intervention.

*(Also verified: `ChurchNoise` is saliency-free, and its CPU and GPU paths are different transforms by design — regional line-partitioned noise versus uniform full-image Gaussian. Since `device_compatible=True`, GPU runs use the uniform variant, so any conclusion about ChurchNoise here is a conclusion about that path.)*

## What the Waterbirds study added

SpuriousBench asked the same headline question in the repair regime — a model already broken by a background shortcut — with ground-truth object masks so that saliency quality could be measured rather than assumed. Full detail in `findings_waterbirds.md`; three points bear directly on the results above.

**The null replicated,** twice: paired median Δ = +0.39pp (95% CI [−0.16, +0.93], ns) under equal compute, and +0.55pp (CI [−0.78, +1.48], ns) at equal deployed epochs, where the two arms selected *different* candidates on half the seeds. So the null is not an artefact of the arms converging on the same augmentation.

**The arbitration criterion is the mechanism.** On Waterbirds the base model sits at ~97% prevalence-weighted accuracy while its worst group is at 59% — the selection criterion and the objective are pulled 38 points apart inside the same model. Imagewoof showed the weaker form of this: the three candidates within 0.7pp of each other, and `edge_ratio` uncorrelated with accuracy. Waterbirds turns "the candidates are close" into "the criterion is measuring the wrong thing".

**The candidates are not rankable by one number.** With masks available, ICD gives the best worst-group accuracy and the worst faithfulness, while AICD gives the best faithfulness on all three mask-grounded metrics and the worst accuracy; under randomised assignment the EBPG separation between them is complete. The Imagewoof accuracy ordering is consistent with this — `icd_only` 33.83 > `churchnoise_only` 33.40 > `aicd_only` 33.14, ICD ahead of AICD, the same direction — but the faithfulness axis on which AICD leads could not be measured here for want of masks. That is precisely why SpuriousBench was built, and it also explains why `edge_ratio`, the stand-in used here, carried no signal.

## Statistical power

With n=10 paired seeds and an observed noise scale of ~0.2pp (SD of the `bnnr_xai − bnnr_random` differences), this design can rule out a *large* XAI-guidance effect (several percentage points) but not a *small* one (sub-percentage-point). The 95% CI of [−0.37, +0.27]pp is the honest statement of what was and was not ruled out — treat it alongside the p-value, not instead of it.

## Conclusions

1. **XAI-guided candidate selection is indistinguishable from random selection** at n=10, with a CI that excludes effects larger than about 0.3pp in either direction.
2. **BNNR's apparent deficit against plain baselines is the epoch split**, not the search mechanism — confirmed by the budget=80 re-run, which closes a 4.46pp gap to 0.09pp.
3. **The arbitration criterion, not the augmentations, is the weak link.** Greedy argmax on validation accuracy has almost nothing to discriminate when candidates sit within 0.7pp of each other.
4. **Accuracy alone is the wrong lens.** Calibration reverses the ranking by an order of magnitude, and the Waterbirds study shows a second axis on which the candidates trade off.
5. **Two library-level defects surfaced:** the rank-biserial formula in `summarize_grand.py`, and the AICD description in `benchmarks/README.md`.

## Suggested follow-up (not part of this task)

**Method.** Condition the intervention on the diagnosis rather than on validation accuracy: the harness already computes saliency-on-object and the group gap, and currently uses them only as a gate. With that in place the three-way search becomes unnecessary, so one diagnosed candidate can train for the full budget — fixing the selection criterion and the epoch split in one change. Concrete proposals are in `findings.md`.

**Measurement.** Promote ECE into the grand-benchmark headline table; it is already recorded and it reverses the ranking. Report a trade-off surface rather than a single winner.

**Still open:** the `--regime pretrained` run (cheap, no code changes, closer to the advertised use case), and a checkpoint-refinement study separating "does search help" from "does search cost more than it buys".

## Protocol caveats (from `benchmarks/README.md`)

- Equal total compute, not equal epochs per deployed model — a deliberately conservative choice for the branch-search thesis.
- The held-out test set is evaluated once per run, at the end, and is never used for selection.
- `bnnr_random` is a literal ablation of `bnnr_xai`: identical compute, identical candidate pool, differing only in the selection mechanism.
- From-scratch, low-data (100 images/class), tight-budget (40 epochs) by design — a harsh setting for a method whose selection signal depends on the model having learned something.
- The budget=80 follow-up changes deployed epochs and baseline maturity together.

## Reproduction

```bash
python benchmarks/run_grand_benchmark.py --dataset imagewoof --device cuda
python benchmarks/summarize_grand.py --results-dir benchmarks/ --datasets imagewoof --markdown
```

Hardware: RTX 5080 Laptop GPU, torch 2.12.1+cu130. Per-run records, git commit and seeds: `benchmarks/results_imagewoof_scratch.json`. Results were produced against bnnr 0.6.5 / commit `0cfa96e`.
