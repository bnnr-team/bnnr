# Attention diagnosis

## What it is

ICD and AICD are opposite operations on attention. ICD masks what the model is already looking at, forcing it to find something else. AICD masks everything else, sharpening what it has. Applying the wrong one is not a small loss — it is an intervention pointed the wrong way.

Which one is right depends on where attention already is, and that is measurable from the saliency maps BNNR computes anyway. `bnnr.analysis.diagnosis` does the measuring and returns a `Diagnosis`: the regime, what it recommends, how confident it is, and every number that produced the answer.

## Why it exists

`selection_mode="xai"` never used saliency to arbitrate. It was a greedy argmax on selection-validation accuracy, and the T20 benchmark found that criterion close to orthogonal to the objective: on Waterbirds it sits at ~97% while the objective sits at 59%, and candidate accuracies differ by fractions of a point. So the selector took an argmax over sub-point differences in a saturated, majority-dominated quantity in order to choose an intervention whose entire purpose is to fix a minority-group failure. It was null at n=10 across two datasets and three protocols.

## The inputs

Everything the rule reads is computed from images and labels alone:

| input | where it comes from |
|---|---|
| `concentration`, `border_mass`, `perturbation_shift` | `bnnr.analysis.saliency_stats` |
| `overall_acc`, `hard_quantile_acc` | the evaluation result (see `hard_quantile_q` in [configuration](configuration.md)) |

**No mask-derived quantity enters the rule.** Not EBPG, not anything needing annotation. That is not an aesthetic preference. BNNR's structural advantage is that it asks for nothing beyond images and labels; a rule that consumes masks stops being deployable and turns BNNR into a different method with a different assumption class. SpuriousBench masks validate this rule from the outside — they never run inside it.

## The rule

Two gates, then a conjunction count.

**Gate 1 — the robustness gap.** With `robustness_gap` at or below `robustness_gap_hi` there is no robustness failure to act on, and the regime is `UNSTRUCTURED` regardless of what the maps look like. A directed intervention needs something to fix.

**Gate 2 — usable structure.** With `concentration` inside the `[concentration_lo, concentration_hi]` band the map is neither diffuse nor concentrated, so it cannot discriminate, and the regime is `UNSTRUCTURED`.

Past both gates, each regime's four-clause conjunction is evaluated:

| regime | clauses | recommends |
|---|---|---|
| `SHORTCUT_SUSPECTED` | diffuse, border-heavy, unstable, real gap | ICD |
| `OBJECT_FOCUSED` | concentrated, central, stable, real gap | AICD |
| `UNSTRUCTURED` | — | ChurchNoise or plain training |

The regime with more satisfied clauses wins. A tie goes to the regime whose weakest clause has the larger margin — the one that is *less nearly false* — so the decision is reproducible rather than dependent on evaluation order.

## Confidence

`confidence` is the fraction of the winning regime's four clauses that hold: one of `0.0`, `0.25`, `0.5`, `0.75`, `1.0`.

It is a count, not a weighted score, and that is the point. A weighted scalar here would be another uncalibrated number of exactly the kind this programme is removing. For `UNSTRUCTURED` it is the fraction of clauses supporting *neither* intervention, so a firm "do nothing" and a borderline one are distinguishable.

`Diagnosis.criteria` carries every clause with its value, its threshold and its margin, so any answer can be reconstructed from the record without re-running anything.

## Thresholds have no defaults

Every threshold in `DiagnosisThresholds` starts at `None`, and `require()` raises `MissingThresholdsError` until they are supplied.

This is deliberate and it is the whole discipline of the exercise. Guessing them now would repeat exactly the mistake that produced `xai_selection_weight` and its preset values of 0.1 and 0.15: numbers nobody measured, shipped as defaults, driving selection for every user. Calibration is a separate, pre-registered study.

| threshold | separates |
|---|---|
| `concentration_lo` / `concentration_hi` | diffuse / no structure / concentrated |
| `border_mass_hi` | central from border-heavy |
| `perturbation_shift_hi` | stable from unstable |
| `robustness_gap_hi` | a gap worth acting on from noise |
| `min_confidence` | acted on from not acted on (not required by the rule itself) |

In a config these live under the `diagnosis:` key, and requesting `selector: diagnosis` with any required one unset fails at construction. See [configuration](configuration.md#diagnosis-thresholds) for the YAML shape and `bnnr.config.load_diagnosis_profile` for loading a calibrated set from file.

**Shadow mode needs none of this.** It records the raw statistics rather than a regime, so it can start collecting calibration samples from every run that was going to happen anyway, at no extra GPU cost, before any threshold exists.

## Using it as a selector

`selector: diagnosis` picks the candidate whose family matches the recommendation, with the selection metric breaking ties *within* that family — the diagnosis says which kind of intervention, not which hyperparameters of it.

Without a diagnosis it selects nothing, with reason `no_diagnosis`. It does not quietly fall back to argmax: a silent fallback would make any benchmark contrast between this selector and argmax measure a blend of the two. An explicit, recorded low-confidence fallback is a policy decision and belongs in the policy layer.

```python
from bnnr.analysis.diagnosis import DiagnosisThresholds, diagnose

diagnosis = diagnose(
    stats,                       # aggregate SaliencyStats for the run
    overall_acc=metrics["accuracy"],
    hard_quantile_acc=metrics["hard_quantile_acc"],
    thresholds=DiagnosisThresholds(
        concentration_lo=...,    # from your calibration, not from us
        concentration_hi=...,
        border_mass_hi=...,
        perturbation_shift_hi=...,
        robustness_gap_hi=...,
    ),
)
print(diagnosis.regime, diagnosis.recommended, diagnosis.confidence)
```

## Known limitation

`perturbation_shift` is optional; when it was not measured the rule reads it as perfectly stable. That pushes the answer towards `OBJECT_FOCUSED` (sharpen what is there) rather than `SHORTCUT_SUSPECTED` (replace it), on the grounds that sharpening a model that did not need it is the cheaper mistake. It is a stated bias, not a neutral default, and a calibration run should supply the statistic rather than rely on it.
