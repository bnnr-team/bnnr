# SpuriousBench — spurious-correlation repair benchmark

Most augmentation benchmarks ask *does this help a model learn?* This one asks a different question: **take a model that is already accurate for the wrong reason, and see whether a method can repair it** — not just recover accuracy, but move the model's attention onto the object it should have been using all along.

That second half is the point. A method that raises worst-group accuracy without moving attention is suspicious; one that moves attention without raising accuracy is cosmetic. Reporting both is what this benchmark is for, and Waterbirds makes it possible because it ships ground-truth object masks.

---

## What it does

1. **Trains a base model to a shortcut state.** ImageNet-pretrained ResNet50, ERM on Waterbirds, where 95% of waterbirds appear on water and 95% of landbirds on land. The model learns the background.
2. **Diagnoses it, and gates on the diagnosis.** Repair proceeds only if the base model actually exhibits the shortcut, defined in advance as: prevalence-weighted accuracy ≥ 0.80 **and** an average-minus-worst-group gap ≥ 15 pp **and** saliency-on-object ≤ 0.55. A model that is not broken is reported as such rather than quietly "repaired".
3. **Runs five repair conditions** from the same diagnosed checkpoint with the same extra budget, and measures both axes.

| Condition | What it does | Role |
|---|---|---|
| `base_frozen` | nothing further | the broken starting point |
| `erm_continue` | keep training, no augmentation | controls for "you just trained longer" |
| `dfr` | retrain the last layer on group-balanced held-out data | the standard baseline (see caveats) |
| `bnnr_random` | BNNR branch search, candidate chosen at random | compute-matched ablation of the XAI guidance |
| `bnnr_xai` | BNNR branch search, candidate chosen by selection-validation | the method under test |

## What it measures

**Robustness.** Worst-group accuracy (the headline number in this literature), the average-minus-worst gap, and a mean accuracy weighted by *training-set* group prevalence — the Sagawa convention, since a plain unweighted mean is the wrong number here.

**Faithfulness**, all against the ground-truth object mask: Energy-Based Pointing Game (fraction of saliency energy on the bird), saliency-mask IoU at 0.5, and Pointing Game (is the peak inside the object). Three metrics rather than one because each is game-able alone — EBPG rewards concentration, IoU rewards coverage, Pointing Game only checks the peak. Agreement across all three is the signal worth trusting.

**Dynamics.** Worst-group accuracy per epoch for every condition that trains, and optionally per-epoch faithfulness on a fixed probe (`--dynamics`).

## Statistics

Paired two-sided Wilcoxon on identical seed sets; Holm-Bonferroni within a contrast family; bootstrap 95% CIs on the paired median difference; matched-pairs rank-biserial; Wilson CIs per group. Exact Wilcoxon p by enumeration for n ≤ 25 when there are no ties and no dropped zeros, otherwise a tie-corrected normal approximation, labelled per contrast. Below six seeds the summarizer suppresses p-values and reports direction and sign-consistency instead, because the minimum achievable p at that sample size is not meaningful.

---

## Running it

```bash
# smoke test: tiny subset, minutes, verifies the whole pipeline
python benchmarks/spurious_repair.py --dataset waterbirds --smoke \
    --device cuda --output ~/spurious_smoke

# the real thing: 5 conditions x 10 seeds
python benchmarks/spurious_repair.py --dataset waterbirds \
    --data-dir ~/data --seeds 0,1,2,3,4,5,6,7,8,9 --faith-cap 500 \
    --device cuda --output ~/bnnr_results/spurious_p3

# summarize
python benchmarks/summarize_spurious.py ~/bnnr_results/spurious_p3 --out report.md
```

**Give every protocol variant its own `--output` directory.** The resume key is `(condition, seed)` and does not encode the budget, the target layer, or the pipeline version, so a run with different settings written into an existing directory will be silently skipped as "already done".

Runs are resume-safe and the results file is written atomically, so a session can be killed and continued. Kill between seeds rather than mid-seed: within a seed all conditions share one base checkpoint, and a half-finished seed would leave its later conditions sitting on a different base after the restart.

**Cost**, measured on an RTX 5080 Laptop at 224 px: roughly 3.9 h per seed for all five conditions, of which about 63% is the faithfulness probe rather than training. The full ten-seed matrix is roughly 39 h, comfortably split across sessions.

## Getting the data

Waterbirds needs two separate downloads: the composited dataset from Stanford, and the CUB segmentation masks, which live in a **different Caltech record** from the main CUB archive (`segmentations.tgz`, record `w9d68-gec53`) and are not inside `CUB_200_2011.tgz`. Without them the faithfulness metrics cannot be computed.

`--download` fetches the Waterbirds archive, but **the CUB mask record currently returns HTTP 403 to a bare urllib request**, so on a fresh machine the masks must be fetched manually and placed alongside the dataset before running with `--data-dir`. This is a known limitation with an open issue, not a design choice.

## Hard ImageNet

The harness is dataset-agnostic and a Hard ImageNet loader is stubbed with the official download instructions. It is not usable yet: Hard ImageNet's public pack ships **masks only**, so it also needs ImageNet-1k access. The loader deliberately raises `NotImplementedError` with instructions rather than guessing the file layout.

---

## Caveats you should read before quoting any number

**The `dfr` condition is a DFR-*style* baseline, not published DFR.** It reaches about 83.4% worst-group accuracy against a published 92.9%. Four identified reasons: the group-balanced subset is not actually balanced to the smallest group (a defect, tracked in an open issue and deliberately left in place here so that this code matches the committed results); no ℓ1 regularisation on the retrained head, which the DFR paper's own ablation prices at roughly 5 pp; no averaging over multiple retrains, roughly 2 pp; and a weaker base feature extractor. Treat it as a reference point for this harness, not as the state of the art.

**The BNNR conditions train the deployed model for `budget // 3` epochs.** Under equal *total* compute, the branch search trains three candidates and deploys one, so the shipped model sees a third of the epochs that `erm_continue` gets. Any BNNR-versus-baseline comparison under equal compute carries this. Running at three times the budget gives equal deployed epochs at three times the compute, and both framings are legitimate — they answer different questions.

**Faithfulness is evaluated at OptiCAM batch size 1.** Batched evaluation is *not* equivalent: OptiCAM optimises its weights with Adam on `-score.mean()`, and the `1/B` gradient scaling cancels everywhere except the epsilon term, leaving the update scaled by `1/(√v̂ + B·eps)`. Measured discrepancy against batch 1 was 0.036–0.047 EBPG, varying between runs because the weight initialisation is random. Correctness was chosen over a 2.6× speedup. `--faith-batch-size` exists for experiments and defaults to 1.

**Worst-group accuracy is quantised.** The worst group has 642 test images, so one image is 0.156 pp and effects below roughly half a percentage point are not resolvable at ten seeds. The summarizer's bootstrap CI is correspondingly lumpy, and quantisation ties push most contrasts onto the approximate Wilcoxon branch.

**One dataset, one architecture, one recipe.** Waterbirds' composited birds have cut-out edges, a confound a natural-image dataset would remove. External validity rests on Hard ImageNet, which is not here yet.

---

## Results in this directory

| File | Contents |
|---|---|
| `results_waterbirds_b15.json` | equal-compute matrix, 5 conditions × 10 seeds |
| `results_waterbirds_b45.json` | equal-deployed-epoch extension, 2 BNNR conditions × 10 seeds |
| `findings_waterbirds.md` | the study: question, method, results, analysis, caveats |

Both files were produced by the version of `spurious_repair.py` in this directory's parent, whose sha256 is recorded in the findings environment section. The summarizer reproduces every number quoted in the findings from these files.

## References

Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts* (ICLR 2020) — the Waterbirds dataset, the worst-group protocol, and the prevalence-weighted mean convention. Kirichenko et al., *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations* (ICLR 2023) — DFR, and the ablations used to account for the gap above. Wang et al. (2020) for the Energy-Based Pointing Game; Moayeri et al. (2022) for the saliency-IoU convention and for Hard ImageNet.
