# BNNR — Master Findings

**Author.** Filip Rusiecki — benchmark design, experiment runs, statistical
analysis and source verification.

## XAI-guided augmentation search across two benchmarks: what works, what does not, and why

**Scope.** This is the combined findings document for T20. It synthesises two studies that are reported in full alongside it: **`findings_imagewoof.md`** — the from-scratch grand benchmark on Imagewoof (10 conditions × 10 seeds, plus an equal-epoch follow-up) — and **`findings_waterbirds.md`** — SpuriousBench, a purpose-built spurious-correlation repair benchmark on Waterbirds (5 conditions × 10 seeds, plus an equal-deployed-epoch extension at n=10 and a candidate-level analysis). Together they cover 190 matrix runs plus follow-ups on one machine, one architecture family, and two regimes — from scratch and repair. Read this document for the argument; read the two companions for the per-study detail, protocol caveats and reproduction commands.

**Provenance and status separation.** Waterbirds numbers were recomputed independently from the raw results JSON by the coordinating chat. Imagewoof numbers are carried from `findings.md` (T20), which recorded them as verified against `results_imagewoof_scratch.json`; they were not re-derived here. Claims about library behaviour are marked *verified-at-source* where the actual `src/bnnr` code was read, and *interpretation* where they are inference. Everything statistical follows the frozen contract: paired two-sided Wilcoxon on identical seeds, Holm within a protocol's own family, bootstrap CIs with widths in prose, exact enumeration only where ties and zeros permit it.

---

# Part I — The short version

Three findings, in descending order of confidence.

**1. XAI-guided candidate selection is indistinguishable from random selection.** From scratch on Imagewoof: median Δ = −0.11 pp, 95% CI [−0.37, +0.27], p = 0.945. In the repair regime on Waterbirds: median Δ = +0.39 pp, 95% CI [−0.16, +0.93], ns. Under equal deployed epochs on Waterbirds: median Δ = +0.55 pp, 95% CI [−0.78, +1.48], ns. Three protocols, two datasets, two regimes, always the same answer at n=10.

**2. The null has a mechanism, and it is the arbitration criterion.** `selection_mode="xai"` is *not* saliency-guided selection — it is a greedy argmax over validation accuracy (*verified-at-source*). On Waterbirds that criterion sits at 97% while the objective sits at 59%; the two are pulled 38 points apart inside the same model. On Imagewoof the three candidates stand within 0.7 pp of each other. So the selector is taking an argmax over a saturated, weakly-related quantity, and half to two-thirds of the time it cannot discriminate at all.

**3. The candidates are not rankable by a single number.** ICD and AICD are opposite operations on the model's attention (*verified-at-source*), and on Waterbirds they trade off: ICD gives the best worst-group accuracy and the worst faithfulness; AICD gives the best faithfulness on all three mask-grounded metrics and the worst accuracy. Under randomised assignment the EBPG separation is complete (Cliff's δ = −1.00, exact Mann-Whitney p = 0.036). Imagewoof showed the same lesson on a different axis: calibration reverses the accuracy ranking by roughly an order of magnitude.

**What this does *not* say.** It does not say BNNR's augmentations fail. Once deployed epochs are matched, BNNR draws level with plain training on Imagewoof (gap 4.46 → 0.09 pp) and moves ~2.6 pp above continued ERM on Waterbirds. The equal-compute deficit was arithmetic, confirmed twice. The weak link is the selection step — and Part IV argues the information needed to fix it is already being computed and thrown away.

---

# Part II — What was built

The second deliverable, alongside the results, is the benchmark itself.

**SpuriousBench** trains a model to a *diagnosed* shortcut state and only then attempts repair. The diagnosis is a pre-registered gate — high average accuracy **and** a large average-minus-worst-group gap **and** low saliency-on-object — so a model that is not actually broken is reported as such rather than quietly "repaired". It fired on 10/10 seeds. The base it produces reproduces the canonical broken model: 59.11% worst-group / ~96.9% weighted mean against Sagawa et al.'s reported ERM at 60.0% / 97.3%, with the harness independently identifying waterbirds-on-land — the 56-image training group — as the worst group on every seed.

It measures two axes at once, because either alone is game-able: **robustness** (worst-group accuracy, average-worst gap, Sagawa prevalence-weighted mean) and **faithfulness** (EBPG + IoU@0.5 + Pointing Game against ground-truth CUB masks). It logs per-epoch trajectories for both. It is resume-safe, atomic-write safe, and it persists the saliency probe arrays so figures can be made without re-running the matrix.

Three things it found about its own instruments, which are contributions in their own right:

- **OptiCAM is batch-size-dependent by construction.** Batched evaluation differs from batch-1 by 0.036–0.047 EBPG, varying run to run. Root cause is analytic: Adam on `-score.mean()` leaves the update scaled by `1/(√v̂ + B·eps)`. We ran everything at batch 1 and forfeited a 2.6× speedup. This affects every batched user of that explainer, not only this benchmark.
- **The pre-registered dynamics metric is degenerate in the repair regime.** "Epochs to 0.80 × final" is crossed at epoch 1 for every condition, because repair starts near its endpoint rather than rising from zero. We report it as it failed and add repair-relative descriptors alongside — the metric design is itself a finding about benchmarking repair.
- **`benchmarks/README.md` misdescribes AICD** as masking "low-saliency *background*". The code masks the least-salient tiles, which equals background only when the model already attends the object — i.e. the description is inverted in exactly the use case the tool is advertised for.

A fourth defect surfaced in the Imagewoof analysis and affects every table the grand-benchmark summarizer has produced: **`summarize_grand.py` computes the rank-biserial effect size as `1 - 2W/(n(n+1))`**, where the matched-pairs statistic `(W⁺ − W⁻)/(W⁺ + W⁻)` requires `1 - 4W/(n(n+1))`. The missing factor of two turns a correlation into the proportion of positive ranks, with range [0.5, 1.0] — so it cannot express a null. The headline Imagewoof contrast printed `r = 0.53` next to `p = 0.945`; recomputed from the raw per-seed accuracies (W⁺ = 17, W⁻ = 19) the true value is **r = −0.06**. Previously published tables convert as `true r = 2 × printed r − 1`. Rows where every seed-pair agrees are unaffected; the distortion is largest exactly where effects are weak. The Waterbirds summarizer written for this project computes the statistic directly and correctly, so its effect sizes stand.

---

# Part III — The evidence

## III.1 The headline, three times

| Study | Protocol | Median Δ (xai − random) | 95% CI | Test |
|---|---|---|---|---|
| T20 Imagewoof, from scratch | equal compute, budget 40 | −0.11 pp | [−0.37, +0.27] | p = 0.945 |
| T20 Imagewoof, from scratch | equal deployed epochs, budget 80 | +0.39 pp | [−0.63, +1.87] | p = 0.461 |
| SpuriousBench Waterbirds, repair | equal compute, budget 15 | +0.39 pp | [−0.16, +0.93] | p ≈ 0.36 |
| SpuriousBench Waterbirds, repair | equal deployed epochs, budget 45 | +0.55 pp | [−0.78, +1.48] | ns |

Per-seed win/loss on Imagewoof was 4/4/2 at budget 40 and 5/3/2 at budget 80. On Waterbirds it was 7+/2−/1 zero at budget 15 and 6+/4− at budget 45. All four point estimates are inside a band narrower than the measurement resolution of the harder benchmark — worst-group accuracy on Waterbirds is quantised at 1/642 = 0.156 pp, giving a detectability floor of roughly half a percentage point at n=10.

**The obvious objection, and why it does not hold.** One could argue the Waterbirds null is trivial because the two arms selected the *same* candidate on 70% of seeds at budget 15 — of course identical training gives identical results. The equal-deployed-epoch extension answers this directly: at budget 45, agreement falls to 50% (and to 38% over the first eight seeds), because `bnnr_xai` stops defaulting to ICD once candidates train longer. **The arms genuinely diverge on half the seeds and the null persists.** So it holds both where selection has nothing to do and where it does something.

**What the disagreement seeds look like.** They are high-variance coin flips, not a systematic edge. The cleanest evidence is a mirror pair: at budget 15, seed 9 (xai→ICD, random→AICD) gives **+5.30 pp**; at budget 45, seed 4 (xai→AICD, random→ICD) gives **−5.61 pp**. Comparable magnitude, opposite sign, mirrored candidate assignment. In the budget-15 matrix, seed 9 alone is doing most of the work pulling the overall median positive.

## III.2 Why: the arbitration criterion is measuring the wrong thing

*Verified-at-source.* `selection_mode="xai"` trains all three candidates from a shared checkpoint and greedily keeps whichever scored highest on **selection-validation accuracy**. Saliency enters BNNR only *inside* ICD and AICD, when deciding which pixels to mask. The contrast both benchmarks run is therefore not "saliency-informed selection versus random" — both arms use saliency-informed augmentations — it is **accuracy-argmax versus random**.

The criterion fails for a structural reason that Waterbirds makes numerically visible. The base model sits at ~97% prevalence-weighted accuracy while its worst group is at 59%; overall accuracy is saturated and dominated by the majority groups, of which landbird-on-land alone is 73% of the training distribution. Observed candidate validation accuracies differ by fractions of a point (seed 0: ICD .8749/.8807, AICD .8816/.8816, ChurchNoise .8549/.8540). So the selector takes an argmax over sub-point differences in a saturated, majority-dominated quantity and uses it to choose an intervention whose purpose is to fix a *minority-group* failure.

Imagewoof showed the same thing in weaker form and revision 1 of T20 already flagged it: the three standalone candidates land within 0.7 pp of each other (`icd_only` 33.83, `churchnoise_only` 33.40, `aicd_only` 33.14), and the `edge_ratio` saliency diagnostic showed no correlation with accuracy (ρ = 0.017 over 100 runs). Waterbirds upgrades "the candidates are close" to "the criterion is nearly orthogonal to the objective".

**This is why more seeds cannot help.** The problem is the choice of criterion, not the sample size.

## III.3 The candidates are opposite interventions, and they optimise different axes

*Verified-at-source:*

```
ICD  — masks the MOST salient tiles   (invert_mask = False; threshold at the 75th percentile)
AICD — masks the LEAST salient tiles  (invert_mask = True;  threshold at the 25th percentile)
ChurchNoise — saliency-free noise; CPU and GPU paths are different transforms by design
```

ICD destroys what the model relies on. AICD destroys what the model ignores. Which is appropriate depends on where the model's attention already is — and on a broken model, that is precisely what is wrong.

Pooled medians over all BNNR runs on Waterbirds (both arms, both protocols):

| candidate | n | worst-group acc | EBPG | IoU@0.5 | Pointing Game |
|---|---|---|---|---|---|
| ICD | 12 | **69.00** | 0.2825 | 0.2555 | 0.5513 |
| AICD | 6 | 64.64 | **0.2956** | **0.2739** | **0.5982** |
| ChurchNoise | 2 | 68.07 | 0.2876 | 0.2673 | 0.5682 |

Two better-controlled cuts of the same data agree.

**Randomised assignment.** In `bnnr_random` the candidate is a verified uniform draw — `random.Random(seed).randint(0,2)` predicts every stored `selected_candidate` for seeds 0–9 exactly — so this is an unconfounded randomised comparison, small but clean. ICD (n=5) versus AICD (n=3): on EBPG, exact Mann-Whitney **p = 0.036** with **Cliff's δ = −1.00**, complete separation — every AICD run scored higher than every ICD run. On worst-group accuracy, p = 0.161, δ = +0.67 favouring ICD.

**Within-seed pairing.** On the three seeds where one arm drew ICD and the other AICD from the same base checkpoint, the only difference being the augmentation: ICD gave higher worst-group accuracy on 2/3 (median +4.52 pp), while **AICD gave higher EBPG, IoU and Pointing Game on 3/3** (medians +0.0090, +0.0195, +0.0315). Agreement across all three faithfulness metrics is exactly the robust signal the trio was designed for, since each is game-able alone.

**Honest limits.** This analysis was not pre-registered; it is reported as exploratory with its randomised cut, its paired cut, and effect sizes rather than as a confirmatory test. `ChurchNoise` at n=2 supports nothing. And the *mechanism* is not settled: on a model attending the background, AICD masks mostly the bird, yet produces the larger faithfulness gain. Candidate explanations — that filling the ignored regions removes the diffuse evidence saliency was spreading over, or that the model re-anchors on whatever structure survives — are not distinguished by these runs.

## III.4 The same lesson on a different axis: calibration

On Imagewoof, expected calibration error reverses the ranking:

| Condition family | Median accuracy | Median ECE |
|---|---|---|
| torchvision policies (RandAugment / TrivialAugment / AutoAugment) | 27.8–31.7% | **0.024–0.042** |
| BNNR (`bnnr_xai` / `bnnr_random`) | 29.5% | 0.054–0.060 |
| no_aug, ICD-only, AICD-only, ChurchNoise-only, ICD+AICD fixed | 33.1–34.0% | **0.230–0.257** |

The most accurate conditions are the worst calibrated by roughly an order of magnitude. Combined with III.3, the general claim both benchmarks support is: **no single scalar ranks these conditions, and which one you optimise decides the winner.** A selection rule that optimises one scalar cannot express that.

## III.5 The equal-compute protocol penalises search, and this is arithmetic

Under equal total compute, the deployed BNNR model trains for a fraction of the budget while single-augmentation conditions train the full budget on one model. On Imagewoof the deployed model saw 20 of 40 epochs; on Waterbirds, `B//3` = 5 of 15. `total_gpu_epochs` is matched; epochs that end up in the shipped model are not.

Tested directly, twice:

| Study | Change | Effect |
|---|---|---|
| T20 Imagewoof | budget 40 → 80 (deployed 20 → 40 epochs) | gap to `no_aug` 4.46 → **0.09 pp** (xai), 4.35 → 0.47 pp (random) |
| SpuriousBench Waterbirds | budget 15 → 45 (deployed 5 → 15 epochs) | from ~0.2–1.2 pp *below* `erm_continue` to ~2.6 pp *above* |

On Waterbirds the within-arm contrast — same arm, same selection rule, only the epoch budget differing — gives +1.95 pp (xai) and +3.35 pp (random), 6/8 positive. Nothing survives Holm at n=8 (best adjusted 0.094), so the statement is that the direction is consistent and the sample is the limit. Note that the Waterbirds extension buys equal deployed epochs at **3× the compute**, and is a diagnostic arm rather than a headline.

**Conclusion.** BNNR's apparent underperformance against plain baselines under equal compute is dominated by the epoch split, not by the search-and-select mechanism. This matters for how the method should be configured (Part IV, Proposal E) and for how future comparisons should be reported.

## III.6 Absolute context, and the assumption asymmetry

Published worst-group accuracies on Waterbirds: ERM 63.7–74.9 · JTT 86.7 · CnC 88.5 · SSA 89.0 · SUBG 89.1 · Group DRO 91.4 · DFR 92.9. Ours: base 59.1 · erm_continue 65.1 · bnnr_random 63.9 · bnnr_xai 64.9 · dfr-style 83.4.

**Every one of our repair conditions falls below the published mitigation range, and the BNNR arms fall far below it.** "BNNR matches continued ERM" is a within-protocol relative statement; it does not mean BNNR is competitive on Waterbirds. Our DFR-style baseline is itself ~9.5 pp below published DFR, for four identified reasons — a group-balancing implementation defect, no ℓ1 regularisation, no averaging over retrains, and a weaker base extractor — three of which the DFR paper's own ablations price.

**The asymmetry runs the other way, and it matters.** These methods consume different supervision: DFR trains its last layer on a group-labelled, group-balanced held-out set; Group DRO uses group labels during training; JTT and CnC use them for tuning. **BNNR uses no group labels anywhere.** It sits in a strictly weaker-assumption class than every method in that table, so the 18 pp gap to DFR is not an apples-to-apples defeat. This is the strongest legitimate framing available and it costs nothing in honesty — but it also defines the constraint that any fix must respect (Part IV).

## III.7 The unexamined dependency: is the saliency any good?

Everything BNNR does downstream of the explainer assumes the saliency is informative. Neither benchmark has established that, and three signals suggest caution.

- `edge_ratio` — the T20 stand-in for saliency quality — showed no correlation with accuracy (ρ = 0.017 over 100 runs; ρ = −0.074 over the 20-run follow-up).
- On Waterbirds, EBPG values cluster in 0.26–0.30 for every condition. **For a perfectly uniform saliency map, EBPG equals the mask's pixel fraction.** Whether 0.27 is meaningfully above that baseline is unknown because the per-image mask-coverage distribution has not been computed. Until it is, the dynamic range of the faithfulness axis is unquantified.
- The intended faithfulness noise floor could not be obtained: the base diagnosis and `base_frozen` were expected to be two independent OptiCAM passes over identical weights, but paired |ΔEBPG| has median 0.0002 and maximum 0.0004, indicating the second reuses cached maps. **We therefore report no floor**, and the support for III.3 rests on complete separation, trio agreement, and 3/3 paired consistency rather than on a noise-threshold argument.

This is the largest open question in the whole programme, and Part IV Proposal D is designed to answer it.

---

# Part IV — How to make `bnnr_xai` actually work

Seven proposals, ordered by expected value per unit of implementation cost. Each is falsifiable on the benchmarks as built.

**A constraint applies throughout, and it is the point of the section.** BNNR's structural advantage is that it asks the user for nothing beyond images and labels (III.6). Every method it is compared against consumes something extra: DFR needs a group-labelled, group-*balanced* held-out set — you must know in advance which groups exist and have enough of each to balance them, which is a much stronger demand than it first sounds; Group DRO needs group labels during training; JTT and CnC need them for tuning. A fix that quietly acquires that supervision does not improve BNNR, it converts BNNR into one of those methods.

**Masks are not a cheap substitute for group labels.** A group label is one integer per image; a segmentation mask is a per-pixel annotation. Anyone who can supply masks can almost certainly supply group labels, so "needs masks, not group labels" is not a weaker assumption — it is a *different* one, and in most settings a heavier one. This document's earlier drafts blurred that, so the proposals are now split by what they actually require:

| tier | what the user must supply | proposals |
|---|---|---|
| **Tier 1 — deployable** | nothing beyond images and labels | A, B, E, F |
| **Tier 2 — research, or mask-rich domains only** | ground-truth object masks | C, D |
| **Tier 3 — reporting** | nothing | G |

**Tier 1 is the main line.** It is what a BNNR user can actually run, and it addresses both defects the two benchmarks identified: a selection criterion that measures the wrong thing, and an epoch split that costs the deployed model two thirds of its budget. Tier 2 is worth doing, but as *studies that tell us how much headroom exists* — and, secondarily, for the domains where masks already exist as a by-product (medical imaging, remote sensing, industrial inspection). Neither Tier 2 proposal should ship as a default feature.

Proposal H turns this table from documentation into behaviour: rather than asking the user which tier applies to them, BNNR detects what supervision the dataset carries and takes the strongest applicable path, with the label-free one as the default and the reference.

## Proposal A — Condition the intervention on the diagnosis, not on validation accuracy **[Tier 1]**

*The cheapest high-value change, and it follows directly from III.2 and III.3.*

The choice between ICD and AICD depends on **where the model is already looking**, because the two are opposite operations on its attention (III.3). A method that picks between them by validation accuracy is answering a different question from the one that determines which is appropriate. So make the diagnosis choose.

The diagnosis has to be built from what BNNR already has, not from masks. **BNNR computes saliency maps for every image as part of its normal operation** — that is what ICD and AICD consume. Those maps can be characterised without knowing where the object is:

- **How concentrated is the saliency?** Entropy or Gini over the map. A model relying on one strong cue produces a peaked map; a model spreading evidence produces a flat one.
- **Where does the mass sit?** The existing `edge_ratio` — the fraction of saliency mass in the outer border — separates "attends the frame or context" from "attends something central". T20 established that `edge_ratio` does not predict accuracy (ρ = 0.017), but it was never tested as a *switch between augmentations*, which is a different and much weaker demand.
- **How stable is it across augmentation?** If saliency moves when the background changes but not when the object is occluded, the model is keying on the background. This needs no annotation at all — only two forward passes.

A concrete, falsifiable rule expressed only in those terms:

- **Diffuse saliency with mass at the border, unstable under background perturbation** → **ICD**. Destroy what the model is leaning on. This is the Waterbirds situation, and ICD is indeed the worst-group-accuracy winner there.
- **Concentrated, centrally located, stable saliency but poor robustness to context shift** → **AICD**. The low-saliency region really is the context for such a model, which is the case the README's description silently assumes.
- **No usable structure in the maps** → **ChurchNoise** or plain training. Saliency-guided masking has nothing to guide it, and forcing it is worse than not using it.

The "poor robustness" term also needs a label-free proxy: accuracy on the highest-loss validation quantile, in the spirit of JTT's group inference. Cost: one branch plus a saliency-statistics helper. Payoff: BNNR stops being "train three, keep whichever scored highest" and becomes a method that diagnoses and then chooses — which is what the name has always promised.

*Note on validation.* On SpuriousBench we can check this rule against ground-truth masks, because they are available there. That is the benchmark's job. The rule itself must not consume them, or it stops being deployable — and this document's earlier draft made exactly that mistake by writing the rule in terms of EBPG.

## Proposal B — Combine ICD and AICD instead of selecting between them **[Tier 1]**

Since III.3 shows they improve *different* axes, a schedule or mixture may dominate either alone on the joint objective: ICD first to break the shortcut, then AICD to concentrate attention; or stochastic per-batch mixing. This is the natural constructive response to a trade-off, and it is cheap — a new candidate type in an existing harness. Note `icd_aicd_fixed` already exists on Imagewoof and landed mid-pack (33.24%), but that is a fixed combination in a regime with no engineered shortcut and no faithfulness measurement; the ordered, diagnosis-aware version on Waterbirds is untested.

## Proposal C — Saliency-regularised last-layer retraining: DFR with a faithfulness penalty **[Tier 2 — needs masks]**

*The strongest novel idea, and the one the DFR authors themselves name as future work.*

Our measurements set up the opportunity precisely. DFR raises worst-group accuracy by ~25 pp while moving EBPG by 0.004 — it *cannot* move attention, because it only reweights a frozen feature extractor. BNNR's augmentations move attention but barely move accuracy. They are complementary by construction: DFR changes how existing features are weighted, BNNR changes which features are learned.

The concrete method: keep DFR's structure — frozen backbone, retrain only the final linear layer — but replace its supervision. For a CAM-style saliency at the last convolutional block, `S(x) = Σ_k w_k A_k(x)` where `A_k` are the frozen feature maps and `w_k` the classifier weights for the predicted class. With the backbone frozen, **EBPG is a differentiable function of `w`**. So train the head with

```
L(w) = CrossEntropy(w) − λ · EBPG(w)
```

on a small reweighting set that carries object masks.

**Be clear about what this costs the user.** It needs masks, and masks are *heavier* supervision than the group labels DFR asks for, not lighter — per-pixel annotation versus one integer per image. So this is not a way to keep BNNR's assumption-class advantage; it is a different method that happens to reuse BNNR's machinery. It is worth building for two reasons: as a **study** of whether attention can be steered from the last layer at all, which is a question the field has not answered; and for the domains where segmentation already exists as a by-product of the workflow — medical imaging, remote sensing, industrial inspection. It should not ship as a BNNR default.

Two implementation notes. First, OptiCAM optimises its weights with Adam and is not a closed-form linear CAM, so use a differentiable surrogate (linear CAM or Grad-CAM) for the training penalty and keep OptiCAM as the frozen evaluation metric — separating the training surrogate from the reported metric is standard and avoids optimising the metric directly. Second, sweep λ and report the trade-off curve rather than a single point; III.3 predicts a frontier, not a winner.

**Falsifiable prediction:** this should beat plain DFR on faithfulness at equal or near-equal worst-group accuracy. If it does not, the claim that attention can be steered from the last layer is wrong, which is itself worth knowing.

## Proposal D — The oracle-mask ceiling: how much is BNNR limited by saliency quality? **[Tier 2 — needs masks]**

*The experiment that bounds everything else, and the answer to III.7. This is a diagnostic run, not a method — nobody would ship it, because if you had the masks you would not need the saliency.*

BNNR masks by saliency. On a broken model the saliency is on the wrong thing — which is exactly why ICD works here (it happens to hit the shortcut) and why AICD masks the bird. If ground-truth masks are available, one can skip the explainer entirely and mask the complement of the object directly.

Run an `oracle_mask` condition on Waterbirds: same harness, same budget, same everything, but the mask comes from the CUB segmentation rather than from OptiCAM. This gives the **ceiling of the augmentation idea under perfect saliency**.

The result is decisive either way. If oracle-mask augmentation still fails to beat `erm_continue`, the limitation is the augmentation approach and no amount of better explanation will rescue it. If it succeeds substantially, then **saliency quality is the binding constraint**, and the lever is the explainer — a completely different research programme from tuning the selector. Cost is one condition, and it is the single most informative run available.

## Proposal E — Stop paying the search tax **[Tier 1]**

III.5 shows the three-way split costs the deployed model two-thirds of its budget, worth roughly 2.6 pp on Waterbirds and 4.4 pp on Imagewoof. If Proposal A works, the search becomes unnecessary: **one diagnosed candidate trained for the full budget B** fixes the selection criterion and the epoch split simultaneously. That combination — A plus E — is the headline recommendation of this document, because it addresses both problems the two benchmarks identified with a single design change.

If some search must be retained, allocate it adaptively: successive halving over the three candidates so that poor branches die after one or two epochs and the winner receives most of the budget, rather than a flat `B/3` each.

## Proposal F — If a selector must be scored, score it on the objective **[Tier 1]**

Worst-group validation accuracy is the obvious criterion and would almost certainly work — but it requires group labels and forfeits III.6. The interesting version is group-label-free: balanced accuracy over an *inferred* partition of the validation set, or accuracy restricted to the highest-loss quantile — the JTT/EIIL family of group-inference ideas applied to *arbitration* rather than to training. This preserves the weaker assumptions while replacing a criterion shown to be nearly orthogonal to the target.

A cleanly separable variant worth running for its scientific value: **select on ΔEBPG** on a masked held-out probe — the "true XAI selection" that `selection_mode="xai"` is not. Our data makes a sharp prediction: on Waterbirds it would systematically choose AICD and would therefore *lose* on worst-group accuracy. Confirming that establishes that selecting for faithfulness and selecting for robustness are different rules with different winners — a more useful result than another attempt to beat random on one axis.

## Proposal G — Report a frontier, not a winner **[Tier 3]**

Given III.3 and III.4, "the best augmentation" is not well defined. The honest deliverable for a library is the trade-off surface over (worst-group accuracy, faithfulness, calibration) together with the diagnosis that indicates where on it a given user should sit. Concretely for the repo: promote ECE into the grand-benchmark headline table — it is already recorded and it reverses the ranking — and ship the diagnosis output as a first-class artifact rather than a gate.

## Proposal H — Let the supervision available decide the path, not the user **[architecture; spans all tiers]**

*Filip's proposal, and the right shape for the tiering above.* The table in this section tells a user which proposals they may use. That is documentation solving a problem the code should solve. BNNR already inspects the dataset before training; it can inspect what supervision came with it and pick the strongest applicable strategy, rather than asking the user to read a compatibility matrix.

```
if ground-truth masks are present:
    diagnose with EBPG / IoU / Pointing Game against the masks;
    choose ICD vs AICD from measured saliency-on-object;
    optionally enable the faithfulness-regularised head (Proposal C)
elif group labels are present:
    diagnose with the average-minus-worst gap;
    choose from measured worst-group behaviour; DFR becomes available as a baseline
else:                                    # the common case
    diagnose from saliency statistics alone — concentration, border mass,
    stability under background perturbation (Proposal A);
    choose ICD vs AICD from those; no extra supervision required
```

**Three things this must get right, or it does more harm than good.**

*The chosen path has to be recorded and reported.* If BNNR behaves differently depending on what the dataset carries, then "BNNR vs baseline" is no longer a single number — two runs on two datasets may be testing two different methods under one name. The selected path belongs in every run record and in every results table, exactly as `selected_candidate` and `fill_strategy` already are. Without that, the benchmark silently stops comparing like with like.

*The label-free path must remain the default and the reference.* It is the one almost every user will hit, it is the only one that preserves BNNR's assumption-class advantage (III.6), and it is what any headline claim about the method should be measured on. The richer paths are enhancements for users who happen to have more, not the thing BNNR is.

*Detection must be strict and must fail loudly.* Partial masks, masks for a subset of classes, group labels that do not correspond to the actual spurious attribute — each of these produces a diagnosis that is confidently wrong, which is worse than no diagnosis. The check should verify coverage and consistency, and fall back to the label-free path with a warning rather than proceeding on incomplete supervision.

**Status: none of the three paths is validated yet.** The mask path rests on the ICD/AICD dissociation, which is exploratory (III.3: n = 5 vs 3 under randomised assignment, at the minimum achievable p). The label-free path has never been tested at all — `edge_ratio` was shown not to predict accuracy, but was never tested as a *switch between augmentations*, which is a weaker and different demand. The group-label path is essentially DFR, of which this project ran only a simplified variant. So this is an architecture worth building toward, and each branch needs its own falsification study before it earns a default. The honest order is: validate the label-free rule first, since it is the one that matters for real users, then the richer paths.

## Infrastructure and documentation fixes (not method changes)

Fix the group-balanced subsampling defect in the DFR baseline. Build a faithfulness noise floor from two *forced-independent* OptiCAM passes. Compute the mask-coverage baseline so the EBPG dynamic range is known (III.7) — this needs no GPU and the artifacts are already on disk. Report the OptiCAM batch-dependence upstream. Correct the AICD description in `benchmarks/README.md` and state, for each augmentation, the condition under which it is the right intervention. Fix the rank-biserial formula in `summarize_grand.py` (`1 - 4W/(n(n+1))`, signed by the median difference) and add a unit test against a known value; every table published with the current formula needs the `2r − 1` conversion.

---

# Part V — What we can and cannot claim

**Can claim.** XAI-guided selection shows no detectable advantage over random at n=10, across two datasets, two regimes and three protocols, with CIs that exclude effects larger than roughly 1 pp on Imagewoof and 1–2 pp on Waterbirds. On Imagewoof the effect size is r = −0.06 (recomputed; the summarizer's printed 0.53 is a formula defect, see Part II). The equal-compute deficit against plain baselines is dominated by the epoch split. ICD and AICD dissociate on the two measured axes, with complete separation on EBPG under randomised assignment. The diagnosed base reproduces the literature's broken ERM model. OptiCAM is batch-size-dependent by construction.

**Cannot claim.** That XAI guidance has *no* effect — only that none is detectable at this power, with a Waterbirds detectability floor near 0.5 pp set by group quantisation. That the ICD/AICD dissociation generalises beyond Waterbirds — it is exploratory, n = 12 versus 6, one dataset, and the mechanism is unexplained. That any faithfulness difference exceeds measurement noise on a noise-threshold argument, since no valid floor exists. That BNNR is competitive with published spurious-correlation methods — it is not, in absolute terms (III.6). Anything about ChurchNoise, at n = 2 and with divergent CPU/GPU paths.

**Generalisation boundary.** One machine, one architecture family (ResNet18 from scratch, ResNet50 pretrained), two datasets, ≤10 seeds, one explainer. Hard ImageNet remains the outstanding external-validity check, and the Waterbirds compositing artefact — birds cut and pasted onto backgrounds — is a known confound that a natural-image dataset would remove.

---

# Part VI — What to run next, in order

1. **Oracle-mask ceiling** (Proposal D). One condition, decisive either way, and it answers the largest open question in III.7. Highest information per GPU-hour of anything on this list.
2. **Mask-coverage baseline for EBPG** (III.7). Zero GPU, artifacts already on disk. Should precede any further quantitative faithfulness claim.
3. **Diagnosis-conditioned selection with full-budget training** (Proposals A + E, both Tier 1). The headline method change, and the only one on this list that ships to users. Testable against the existing matrices with no new baselines needed. Note the dependency: A's decision rule has to be written in terms of saliency statistics BNNR already computes — concentration, border mass, stability under perturbation — never in terms of mask-derived quantities like EBPG, or it stops being deployable. SpuriousBench's masks are for *validating* the rule, not for running it.
4. **T21 fill-strategy ablation** on both datasets, already briefed — with the pre-registered prediction that the best fill is *opposite* across the two, since one is removing a shortcut and the other is regularising.
5. **Saliency-regularised DFR** (Proposal C). The novel-method study; largest scientific upside, largest implementation cost.
6. **Supervision-aware path selection** (Proposal H). Worth designing early so the run record carries the chosen path from the start, but it should only be switched on per branch as each branch is validated — an unvalidated automatic path is worse than an explicit flag.
7. **Hard ImageNet** for external validity, whenever ImageNet access allows.

---

# Appendix — Verified facts, for anyone building on this

*Read from `src/bnnr` at source, not from documentation or memory:*

- `ICD` masks the most-salient tiles (`invert_mask=False`, 75th percentile); `AICD` masks the least-salient tiles (`invert_mask=True`, 25th percentile). `_VALID_FILL_STRATEGIES = {gaussian_blur (default), local_mean, global_mean, noise, solid}`; kwarg is `fill_strategy`; `solid` pairs with `fill_value`; `mask_value` is deprecated and maps to `solid`.
- `ChurchNoise` is saliency-free and `device_compatible=True`; its CPU path adds line-partitioned regional noise while its GPU path adds uniform full-image Gaussian noise — different transforms by design, so GPU runs use the uniform variant.
- `selection_mode="xai"` is a greedy argmax on selection-validation accuracy. No saliency enters the arbitration.
- `XAICache.save_map` persists only index-keyed maps; hash-keyed persistence was removed. Feeding a loader without sample indices silently produces a dead cache and online recomputation every batch.
- Real augmentations operate on unnormalised uint8; `AugmentationRunner._tensor_to_uint8` clips inputs below −0.01 to [0,1], so feeding a normalised batch silently destroys the image rather than raising.
- `benchmarks/README.md` line 64 describes AICD as masking "low-saliency background" — accurate only for a model already attending the object.
- `summarize_grand.py` computes rank-biserial as `1 - 2W/(n(n+1))`; the correct matched-pairs form is `1 - 4W/(n(n+1))`. Printed values have range [0.5, 1.0] and cannot represent a null. Conversion: `true r = 2 × printed r − 1`.
- Grand-benchmark resume key is `(condition, seed, regime)`; SpuriousBench's is `(condition, seed)`. Neither includes budget, fill strategy, target layer or pipeline version, so any protocol variant must use its own results directory.

*Literature anchors, read in full:* Sagawa et al. 2019 — Waterbirds construction (4795 train, smallest group 56), ERM 60.0 worst / 97.3 average, Group DRO 84.6 (strong ℓ2) / 91.4 (grid-searched), and the prevalence-weighted mean convention this benchmark adopts. Kirichenko et al. 2022 — DFR 92.9 ± 0.2 worst-group, ERM base 74.9, the ℓ1 and retrain-averaging ablations used in III.6, the Group-Info column that defines the assumption asymmetry, and the future-work suggestion of saliency- or mask-based supervision for the last layer that Proposal C implements.
