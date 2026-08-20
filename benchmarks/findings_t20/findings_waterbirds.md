# SpuriousBench — Waterbirds findings

**Status:** complete for the Waterbirds dataset. Equal-compute matrix (n=10) and equal-deployed-epoch extension (n=10) are final; the dynamics-trajectory subset (P6 figure material) is outstanding and does not affect any conclusion below.
**Harness:** `benchmarks/spurious_repair.py` sha256 `1d063ac9…`, `benchmarks/summarize_spurious.py` sha256 `c8ec9734…`, tests `aef14919…`. bnnr 0.6.5, repo `0cfa96e`, branch `spurious-repair-benchmark`.
**Environment:** WSL2, RTX 5080 Laptop (sm_120), torch 2.12.1+cu130, Python 3.14.4. All numbers recomputed independently from the raw results JSON, not taken from the summarizer alone.

---

## 1. Question

Take a model that is accurate for the wrong reason — it exploits the image background rather than the bird. Do BNNR's XAI-guided augmentations repair it, and does **XAI-guided candidate selection** repair better than **random** selection?

Falsifiable headline: at n=10 paired seeds, is there a detectable `bnnr_xai` − `bnnr_random` difference in worst-group accuracy?

## 2. Method

**Data.** Waterbirds (`waterbird_complete95_forest2water2`) with CUB segmentation masks (Caltech record w9d68-gec53). Test groups: landbird-land 2255, landbird-water 2255, waterbird-land 642, waterbird-water 642. Training group sizes (Sagawa et al.): 3498 / 184 / 56 / 1057 — the smallest group, waterbirds-on-land, has 56 training images.

**Base model and diagnosis gate.** ImageNet-pretrained ResNet50, ERM for 15 epochs. Pre-registered gate: proceed only if prevalence-weighted accuracy ≥ 0.80 **and** (average − worst) gap ≥ 15 pp **and** EBPG ≤ 0.55. Fired `is_broken=True` on all 10 seeds (worst-group 57.2–61.4%, gaps 35–39 pp). The harness independently identified `waterbird_land` as the worst group on every seed.

**Conditions**, all from the same diagnosed base checkpoint within a seed, all receiving the same extra budget B=15: `base_frozen` (no further training), `erm_continue` (more ERM), `dfr` (last-layer retrain on group-balanced held-out data), `bnnr_random`, `bnnr_xai`. The BNNR arms train three candidates — ICD, AICD, ChurchNoise — for `B//3 = 5` epochs each and deploy one.

**Pipeline.** Augmentations act on unnormalised [0,1] tensors; normalisation happens at a single point after augmentation, shared by every training and evaluation path (`pipeline_version = p2-norm-after-aug`). Target layer `model.layer4` throughout.

**Faithfulness.** EBPG + IoU@0.5 + Pointing Game against ground-truth masks, on normalised in-distribution inputs, 500 masks per group (2000-image probe), OptiCAM at batch size 1. Batched OptiCAM was implemented, tested and **rejected**: it is batch-size-dependent by construction (Adam on `-score.mean()` leaves the update scaled by `1/(√v̂ + B·eps)`), giving EBPG discrepancies of 0.036–0.047 against batch=1. Correctness was chosen over a 2.6× speedup.

**Statistics.** Paired two-sided Wilcoxon on identical seeds; Holm-Bonferroni within each protocol's own contrast family (never pooled across protocols); bootstrap 95% CI on the paired median difference with the width stated; matched-pairs rank-biserial; Sagawa prevalence-weighted mean (weights from the training distribution, per their Appendix C.1); exact Wilcoxon by enumeration when n ≤ 25 with no ties or zeros, otherwise the tie-corrected normal approximation, labelled per contrast.

---

## 3. Results — equal compute (budget = 15, n = 10)

**Worst-group accuracy, medians:** base_frozen 59.11 · erm_continue 65.11 · **dfr 83.41** · bnnr_random 63.86 · bnnr_xai 64.88.

### 3.1 Headline: no detectable difference between XAI-guided and random selection

| | value |
|---|---|
| paired median Δ (xai − random) | **+0.39 pp** |
| 95% CI on the paired median | **[−0.16, +0.93] pp** (width ≈ 1.1 pp) |
| rank-biserial r | +0.378 |
| Wilcoxon two-sided | p ≈ 0.36 (`approx` — a zero and ties in \|d\| disqualify enumeration) |
| Holm-adjusted | 0.68 |

The ten paired differences (pp): +0.47, +0.47, −0.62, +0.31, +1.40, +0.47, −2.49, 0.00, +0.31, +5.30 — seven positive, two negative, one exact zero.

**There is no detectable difference between XAI-guided and random candidate selection at n=10.** The whole contrast lives inside a ±1 pp band. Worst-group accuracy is quantised at 1/642 = 0.156 pp, so the detectability floor is roughly half a percentage point; effects below that cannot be resolved at this seed count, and the bootstrap CI is correspondingly lumpy. This extends T20's from-scratch null (Δ = −0.11 pp, p = 0.945) into the repair regime the method is advertised for.

### 3.2 The only strong effect is the DFR-style baseline

`dfr` beats every other condition on every one of the ten seeds: +17.99 pp over `erm_continue` (r = +1.000, exact p = 0.0020, Holm = 0.0098); `bnnr_xai` sits 18.22 pp below it (r = −1.000). It is the only contrast surviving Holm.

**This is a DFR-*style* baseline, not published DFR.** Ours reaches 83.4% (range 82.1–84.4) against a published 92.9 ± 0.2%. The 9.5 pp shortfall has four identified causes, three of them priced by the DFR paper's own ablations:
1. **Our group-balanced subset is not in fact balanced** — an implementation defect. `group_balanced_subset` takes the first `per_group` items, so with 200 per group against validation groups of 467/466/133/133 the subset is 200/200/133/133, over-weighting the majority groups. Kirichenko et al. subsample every group down to the smallest; their Table 12 prices this axis at roughly 3.7 pp.
2. **No ℓ1 regularisation** on the retrained head — their ablation reports 87.72% without it versus 92.9% with it, roughly 5 pp.
3. **No averaging over multiple retrains** — their Table 10 reports 91.21% for one retrain versus 93.13% for ten, roughly 2 pp.
4. **A weaker base feature extractor** — 15 epochs without augmentation against their 100 augmented epochs.

Item 1 is a genuine bug and is queued as a post-matrix fix; it was not corrected mid-matrix because the harness was frozen and all 50 records used this implementation.

### 3.3 BNNR matches plain continued training despite one third of the deployed epochs

`bnnr_xai` vs `erm_continue`: median −0.23 pp (Holm 0.68, ns). `bnnr_random` vs `erm_continue`: median −1.17 pp (Holm 0.58, ns). Both BNNR arms are statistically indistinguishable from simply training longer — while the deployed BNNR model trained for 5 epochs against `erm_continue`'s 15, because the equal-compute budget is split three ways across candidates. In T20 this same split cost BNNR about 4.5 pp; here the deficit is inside noise.

### 3.4 Dynamics

The pre-registered epochs-to-threshold metric is **degenerate in the repair regime**: because repair starts near its endpoint, the "0.80 × final" threshold is crossed at epoch 1 for every condition, and the absolute 70% threshold censors the entire BNNR pair. This is a reportable finding about dynamics-metric design, not an implementation failure — the metric was defined for curves rising from near zero.

Supplementary descriptors (defined after the pilot exposed the degeneracy and before the remaining seeds ran, motivated by curve shape rather than by any contrast). Median retention, (endpoint − base) / (peak − base): dfr 1.000 · **bnnr_xai 0.808** · erm_continue 0.704 · bnnr_random 0.676. BNNR-xai retains more of its peak than continued ERM does; the single-seed impression from the pilot that the BNNR arms collapse after peaking did not generalise.

---

## 4. Results — equal deployed epochs (budget = 45, n = 10)

**Design and disclosure.** This arm was pre-registered on 2026-07-23, before the main matrix completed, precisely so it could not read as a post-hoc rescue. It was designed at n=6, executed at n=8, then extended to n=10 for seed-set parity with the main matrix **after** the n=8 results were seen. The n=10 p-values are therefore descriptive rather than confirmatory, and the conclusion drawn does not depend on them. Holm is applied within this arm's own family. The arm buys equal *deployed* epochs at **3× the compute** (three candidates × 15 epochs against `erm_continue`'s 15) — it is not compute-matched, and every reading must carry that.

### 4.1 The headline null replicates, under genuinely divergent selection

xai(45) vs random(45), n=10: paired median **+0.55 pp**, six positive / four negative, bootstrap 95% CI **[−0.78, +1.48] pp** (width 2.26), Wilcoxon ns (`approx`; ties in \|d\|).

More importantly, **candidate agreement fell from 7/10 (70%) at budget 15 to 5/10 (50%) at budget 45** — and to 3/8 over the first eight seeds. At the longer budget `bnnr_xai` stops defaulting to ICD. So the null holds both where the two arms mostly train the *same* candidate and where they mostly train *different* ones. This rules out the obvious objection that the equal-compute null was an artefact of the arms converging on the same augmentation.

The disagreement seeds are high-variance rather than systematically favourable to XAI, and the cleanest evidence is a mirror pair: main-matrix seed 9 (xai→ICD, random→AICD) gives **+5.30 pp**, extension seed 4 (xai→AICD, random→ICD) gives **−5.61 pp** — comparable magnitude, opposite sign, mirrored candidate assignment.

### 4.2 The equal-compute deficit was the epoch split (directional, n=8 figures)

`bnnr_xai(45)` vs `erm_continue(15)`: median **+2.65 pp**, 7 positive / 1 negative, raw p = 0.023. `bnnr_random(45)` vs `erm_continue(15)`: median **+2.57 pp**, 6/2, raw p = 0.109. Under equal compute both arms sat 0.2–1.2 pp *below* `erm_continue`; matching deployed epochs moves them roughly 2.6 pp above it. **After Holm over this arm's four-contrast family nothing reaches significance** (best adjusted p = 0.094). The direction is consistent; the sample is the limit.

Mechanistically the cleanest contrast is within-arm, where only the epoch budget differs: `bnnr_xai(45) − bnnr_xai(15)` median +1.95 pp (6/2, raw p = 0.078) and `bnnr_random(45) − bnnr_random(15)` median +3.35 pp (6/2, raw p = 0.039). Both point the same way as the primary. This mirrors T20's budget 40→80 result: the deficit was the truncated deployed training, not the augmentation.

**Base-drift control.** The two protocols retrain the base independently, so cross-protocol pairing shares seed and recipe but not the checkpoint. Measured drift in base worst-group accuracy: median −0.003 pp, maximum 1.24 pp (seed 4) — small against the +2.6 pp effect, so the pairing is defensible. Seed 4 carries both the largest drift and the large negative in §4.1 and is flagged accordingly.

---

## 5. External validation and absolute context

**Our diagnosed base reproduces the canonical broken model.** Ours: 59.11% worst-group / ~96.9% weighted mean. Sagawa et al. report ERM at 60.0% / 97.3%. The starting point is the literature's broken model, independently reproduced, not merely one our own gate declared broken. (Kirichenko et al.'s stronger ERM base, 74.9%, comes from 100 augmented epochs against our 15 unaugmented ones.)

**Where our numbers sit in the field.** Published Waterbirds worst-group accuracies: ERM 63.7–74.9 · JTT 86.7 · CnC 88.5 · SSA 89.0 · SUBG 89.1 · Group DRO 91.4 · DFR 92.9. Ours: base 59.1 · erm_continue 65.1 · bnnr_random 63.9 · bnnr_xai 64.9 · dfr-style 83.4. **Every one of our repair conditions falls below the published mitigation range, and the BNNR arms fall far below it.** "BNNR matches continued ERM" is a within-protocol relative statement and must be read against these absolute numbers; it does not mean BNNR is competitive on Waterbirds.

*(Note on a figure that circulated in our own planning: the "~84 ceiling" is Group DRO under a strong ℓ2 penalty in Sagawa's Table 1, not DFR. Group DRO with their full grid search reaches 91.4.)*

**The assumption asymmetry, which runs the other way.** Kirichenko et al. carry a "Group Info" column precisely because these methods consume different supervision: DFR trains its last layer on a group-labelled, group-balanced held-out set; Group DRO uses group labels during training; JTT and CnC use them for tuning. **BNNR uses no group labels anywhere** — its arbitration runs on plain validation accuracy. It therefore sits in a strictly weaker-assumption class than every method in that table, and the 18 pp gap to DFR is not an apples-to-apples defeat. This must be stated in the same breath as the gap.

---

## 6. Analysis — why XAI-guided selection did not beat random

This section is the substantive contribution beyond the null itself.

### 6.1 The arbitration is not XAI-guided

Verified by reading the library, not from memory: `selection_mode="xai"` performs a **greedy best-of-three on selection-validation accuracy**. Saliency enters BNNR only *inside* ICD and AICD, when deciding which pixels to mask. The contrast we ran is therefore not "saliency-informed selection versus random selection" — both arms use saliency-informed augmentations. It is **accuracy-argmax selection versus random selection**. The name invites a stronger reading than the mechanism supports, and any interpretation of the null has to start here.

### 6.2 The arbitration criterion is nearly orthogonal to the objective

On Waterbirds the base model already sits at ~97% prevalence-weighted accuracy while its worst group is at 59%. Overall accuracy is saturated and dominated by the majority groups — landbird-on-land alone is 73% of the training distribution. Observed candidate validation accuracies differ by fractions of a percentage point (for example on seed 0: ICD .8749/.8807, AICD .8816/.8816, ChurchNoise .8549/.8540).

So the selection rule takes an argmax over sub-point differences in a saturated, majority-dominated quantity, and uses it to choose between interventions whose purpose is to fix a minority-group failure. **This is the precise sense in which accuracy is the wrong measure here**, and Waterbirds makes it visible because the two quantities are numerically pulled apart by ~38 points in the same model.

Consequences follow directly. Where the criterion cannot discriminate, both arms pick the same candidate — 70% of seeds at equal compute. Where it does discriminate, it is discriminating on something weakly related to the target, so the outcome swings hard in both directions: the s9/s4 mirror pair, +5.30 and −5.61 pp.

### 6.3 The finding: ICD and AICD are opposite interventions, and they optimise different axes

Verified at source:

```
ICD  — masks the MOST salient tiles   (invert_mask = False; threshold at the 75th percentile)
AICD — masks the LEAST salient tiles  (invert_mask = True;  threshold at the 25th percentile)
ChurchNoise — saliency-free noise; on GPU the uniform full-image Gaussian path
```

ICD destroys what the model currently relies on. AICD destroys what the model currently ignores. These are not two flavours of the same augmentation; they are opposite operations on the model's own attention, and which is appropriate depends on where that attention already is.

**They dissociate empirically, and the dissociation replicates in both arms.**

Between-run medians, pooled across both arms and both protocols:

| candidate | n | worst-group acc | EBPG | IoU@0.5 | Pointing Game |
|---|---|---|---|---|---|
| ICD | 12 | **69.00** | 0.2825 | 0.2555 | 0.5513 |
| AICD | 6 | 64.64 | **0.2956** | **0.2739** | **0.5982** |
| ChurchNoise | 2 | 68.07 | 0.2876 | 0.2673 | 0.5682 |

**ICD gives the best worst-group accuracy and the worst faithfulness. AICD gives the best faithfulness on all three metrics and the worst worst-group accuracy.**

Two independent, better-controlled cuts of the same data agree:

*Randomised assignment.* In `bnnr_random` the candidate is a verified uniform draw — `random.Random(seed).randint(0,2)` predicts every stored `selected_candidate` for seeds 0–9 exactly — so this is an unconfounded, if small, randomised comparison. ICD (n=5) versus AICD (n=3): on EBPG, exact Mann-Whitney **p = 0.036** with **Cliff's delta = −1.00**, i.e. *complete separation* — every AICD run scored higher than every ICD run. On worst-group accuracy, p = 0.161, Cliff's delta = +0.67 in ICD's favour.

*Within-seed pairing.* On the three seeds where one arm drew ICD and the other AICD from the same base checkpoint, the only difference being the augmentation: ICD gave higher worst-group accuracy on 2 of 3 (median +4.52 pp), while **AICD gave higher EBPG on 3 of 3 (median +0.0090), higher IoU on 3 of 3 (+0.0195) and higher Pointing Game on 3 of 3 (+0.0315)**. Agreement across all three faithfulness metrics is exactly the robust signal the trio was designed to detect, since each is game-able alone.

This is the trade-off the benchmark's own methodology anticipated — that a method raising worst-group accuracy without moving attention is suspicious, and one moving attention without raising accuracy is cosmetic. We now have an instance of each, plus the extreme case in `dfr`, which raises worst-group accuracy by 25 pp while moving EBPG by 0.004.

**An honest note on mechanism.** The direction is not the naive one. On a model attending the background, ICD masks mostly background — the shortcut — and indeed yields the larger robustness gain. AICD masks mostly what the model ignores, which on this model includes much of the bird, and yet produces the larger *faithfulness* gain. Our data establishes the dissociation; it does not settle why AICD concentrates attention onto the object. Candidate explanations — that filling the ignored regions removes the diffuse evidence saliency was spreading over, or that the model re-anchors on whatever structure survives — are not distinguished by these runs. We report the effect and flag the mechanism as open.

### 6.4 Two measurement caveats that bound §6.3

**The faithfulness noise floor is unavailable from this matrix.** The design intended to obtain one for free by comparing the base diagnosis against `base_frozen` — identical weights, nominally two independent OptiCAM passes. The data refute the assumption: paired |ΔEBPG| has a median of 0.0002 and a maximum of 0.0004, far too small for independent runs of a randomly-initialised optimiser, indicating that `base_frozen` reuses the cached diagnosis maps. We therefore **do not report a floor**, and we do not claim that a 0.013 EBPG difference exceeds measurement noise on that basis. The support for §6.3 comes instead from complete separation under randomised assignment, agreement across three metrics, and 3/3 consistency in the paired cut. A proper floor requires a design that forces two independent OptiCAM passes and is queued.

**The dynamic range of EBPG on this dataset is unquantified.** All EBPG values cluster in 0.26–0.30. For a perfectly uniform saliency map, EBPG equals the mask's pixel fraction, so the informative question is how far these values sit above that baseline — which requires the per-image mask-coverage distribution. That distribution is computable at zero GPU cost from the persisted saliency artefacts and `_probe_reference.npz`, and should be computed before the faithfulness axis is interpreted quantitatively in any follow-up.

### 6.5 Why "select by validation accuracy" is wrong in principle, not just underpowered

The dissociation means **there is no single best candidate**. ICD and AICD occupy different points on a robustness/faithfulness trade-off; which one a user wants depends on their objective and on the diagnosis of their model. A scalar argmax silently commits to one criterion — and the criterion currently chosen is neither of the two objectives the benchmark measures. Adding seeds would not fix this. The problem is the choice of criterion, not the sample size.

---

## 7. Recommendations

Ranked by expected value against implementation cost. All are testable on SpuriousBench as built.

**1. Condition the intervention on the diagnosis rather than on validation accuracy.** The harness already computes, before any repair, everything needed: EBPG (where attention is), the average-minus-worst gap (how bad the group failure is), and per-group accuracies. A falsifiable rule follows directly from §6.3 — attention off the object plus a large group gap calls for ICD, since the region to destroy is the one the model relies on; attention already on the object but poor robustness to context calls for AICD, since the region to destroy is the surrounding context; no usable saliency signal falls back to a saliency-free perturbation. This is a genuine method contribution, it costs one extra branch, and it converts BNNR's diagnostic output — currently used only as a gate — into the thing that steers the repair.

**2. Align the selection criterion with the objective, without acquiring group labels.** Worst-group validation accuracy would be the obvious criterion, but using it would move BNNR into DFR's and Group DRO's assumption class and forfeit its main structural advantage (§5). The interesting version is group-label-free: select on balanced accuracy over an inferred partition of the validation set, or on accuracy restricted to the highest-loss quantile — the JTT/EIIL family of group-inference ideas applied to arbitration rather than to training. This preserves BNNR's weaker assumptions while replacing a criterion we have shown to be nearly orthogonal to the target.

**3. Faithfulness-guided selection — and report it as a different rule, not a better one.** Selecting the candidate with the largest ΔEBPG on a masked held-out probe is the natural "true XAI selection" that the current `selection_mode="xai"` is not. Our data makes a sharp prediction: on Waterbirds it would systematically choose AICD and would therefore *lose* on worst-group accuracy. That prediction is the experiment. Confirming it would establish that selecting for faithfulness and selecting for robustness are genuinely different rules with different winners — a stronger and more useful result than another attempt to beat random on a single axis. Kirichenko et al. name mask- and saliency-based supervision as future work in this literature, so the direction is not idiosyncratic to us.

**4. Stop selecting; combine.** Because ICD and AICD improve different axes, a schedule or mixture — ICD first to break the shortcut, AICD after to concentrate attention, or stochastic per-batch mixing — may dominate either alone on the joint objective. This is cheap: it is a new candidate type in an existing harness, and it is the natural constructive response to a trade-off.

**5. Report the Pareto front instead of a winner.** Given the dissociation, a single "best augmentation" recommendation is not well defined. The honest deliverable for a library is the trade-off surface over (worst-group accuracy, faithfulness) with the diagnosis that indicates where on it a given user should sit.

**Also queued, in the harness rather than the method:** fix the group-balanced subsampling defect in the DFR baseline (§3.2 item 1); build a faithfulness noise floor from two forced-independent OptiCAM passes; compute the mask-coverage baseline for EBPG (§6.4); and report the OptiCAM batch-dependence upstream, since it affects every batched user of that explainer, not only this benchmark.

---

## 8. Caveats

- **The equal-compute epoch split.** Under budget B the deployed BNNR model trains B//3 epochs against `erm_continue`'s B. Every equal-compute BNNR-versus-baseline line is subject to this; §4 is the control for it, at 3× compute.
- **DFR wording.** A DFR-*style* last-layer retrain, ~9.5 pp below published DFR, with the four causes enumerated in §3.2.
- **Faithfulness probe.** 500 masks per group at OptiCAM batch size 1; batched OptiCAM tested and rejected as batch-size-dependent.
- **Resolution.** Worst-group accuracy is quantised at 0.156 pp (642 test images); the detectability floor is roughly 0.5 pp at n=10, and quantisation ties push most contrasts onto the approximate Wilcoxon branch.
- **P3b sequence.** Designed at n=6, executed at n=8, extended to n=10 after seeing n=8; p-values there are descriptive.
- **Section 6.3 is exploratory.** The candidate-level comparison was not pre-registered; it is reported with its randomised-assignment cut, its within-seed cut, and small-n effect sizes rather than as a confirmatory test. `ChurchNoise` (n=2, and on GPU a different transform from its CPU path) supports essentially no conclusion.
- **The pilot records** from the earlier conv3/old-pipeline run are quarantined and never mixed with these matrices.
- **Single dataset, single architecture, single recipe.** External validity rests on Hard ImageNet.
- **Endpoint is the deployed model** — no early stopping, no best-epoch selection. All arms ship below their own peak, and the metric understates short-budget arms proportionally more.

---

## 9. Bottom line

On Waterbirds, in the repair regime, XAI-guided candidate selection shows **no detectable advantage over random selection at n=10**, and the null replicates under a second protocol in which the two arms select different candidates half the time. This extends T20's from-scratch null into the setting the method is advertised for.

The null has a mechanism. The arbitration is not saliency-guided at all — it is a greedy argmax on validation accuracy, a quantity that on this dataset is saturated at 97% while the objective sits at 59%. Half to two thirds of the time it cannot discriminate between candidates; when it does, it is discriminating on something weakly related to the target, and the outcome is a coin flip with a ±5 pp swing.

Underneath that, the candidates themselves are not interchangeable and not rankable by a single number. ICD and AICD are opposite interventions on the model's attention, and in our data they trade off: ICD buys worst-group accuracy, AICD buys saliency-on-object, with complete separation on EBPG under randomised assignment. A benchmark that reports only accuracy would have missed this entirely, and an arbitration rule that optimises only accuracy cannot express it.

The constructive reading is therefore not that BNNR's augmentations fail — both arms match plain continued training at one third of the deployed epochs, and the equal-epoch extension shows the deficit was the split rather than the method. It is that **the selection step is the weak link, and the information needed to fix it is already being computed and discarded**: the diagnosis that gates the repair should also choose it.
