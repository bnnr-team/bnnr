"""Which intervention the attention evidence points at, and how strongly.

``selection_mode="xai"`` never used saliency to arbitrate. It was a greedy
argmax on selection-validation accuracy, and T20 found that criterion close to
orthogonal to the objective: on Waterbirds it sits at ~97% while the objective
sits at 59%, and candidate accuracies differ by fractions of a point. So the
selector took an argmax over sub-point differences in a saturated,
majority-dominated quantity in order to choose an intervention whose entire
purpose is to fix a minority-group failure.

ICD and AICD are opposite operations on attention. ICD masks what the model is
already looking at, forcing it to find something else; AICD masks everything
else, sharpening what it has. Which one is right depends on where attention
already is, and that is measurable. This module does the measuring.

**Everything here is computed from images and labels alone.** No EBPG, no
mask-derived quantity, nothing that needs annotation. That is not an aesthetic
preference: a rule that consumes masks stops being deployable and turns BNNR
into a different method with a different assumption class. SpuriousBench masks
validate this rule from the outside; they never run inside it.

**No threshold has a default.** Guessing them now would repeat exactly the
mistake that produced ``xai_selection_weight`` and its preset values of 0.1 and
0.15: numbers nobody measured, shipped as defaults, driving selection for every
user. :class:`DiagnosisThresholds` starts fully ``None`` and
:meth:`DiagnosisThresholds.require` refuses to proceed without them. Calibration
is FIX-7-2; shadow mode (FIX-3-2) records the raw statistics and needs none of
this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bnnr.analysis.saliency_stats import SaliencyStats

__all__ = [
    "AttentionRegime",
    "Diagnosis",
    "DiagnosisThresholds",
    "MissingThresholdsError",
    "diagnose",
]

#: Doc the error message points at, so the fix is one click from the traceback.
_CALIBRATION_DOC = "docs/diagnosis.md"


class AttentionRegime(str, Enum):
    """Where the model's attention already is, in the terms the rule decides on."""

    #: Diffuse, drawn to the border, unstable under perturbation, with a real
    #: robustness gap. The picture of a model reading context rather than object.
    SHORTCUT_SUSPECTED = "shortcut_suspected"

    #: Concentrated, central, stable, and still failing its hard quantile. The
    #: model has found the object and needs the signal sharpened, not replaced.
    OBJECT_FOCUSED = "object_focused"

    #: The maps carry no usable structure, or there is no robustness gap worth
    #: intervening on. Neither directed intervention is indicated.
    UNSTRUCTURED = "unstructured"


class MissingThresholdsError(RuntimeError):
    """Diagnostic mode was requested before its thresholds were calibrated."""


@dataclass(frozen=True)
class DiagnosisThresholds:
    """Cut points for the decision rule. All ``None`` until calibrated.

    ``concentration_lo`` / ``concentration_hi`` bracket a middle band. A map
    inside it is neither diffuse nor concentrated, which is the definition of
    "no usable structure" this module uses.

    The remaining three are one-sided: above ``border_mass_hi`` the mass sits at
    the frame, above ``perturbation_shift_hi`` the explanation does not survive
    its own perturbation, and above ``robustness_gap_hi`` there is a robustness
    failure worth acting on.
    """

    concentration_lo: float | None = None
    concentration_hi: float | None = None
    border_mass_hi: float | None = None
    perturbation_shift_hi: float | None = None
    robustness_gap_hi: float | None = None

    #: Diagnoses at or below this confidence are not acted on. FIX-4-2 falls
    #: back to ``metric_argmax`` there.
    min_confidence: float | None = None

    #: Field names the rule cannot run without. ``min_confidence`` is not among
    #: them: a caller may want the regime and its confidence without a policy
    #: for what to do below the line.
    REQUIRED = (
        "concentration_lo",
        "concentration_hi",
        "border_mass_hi",
        "perturbation_shift_hi",
        "robustness_gap_hi",
    )

    def missing(self) -> tuple[str, ...]:
        """Required thresholds that are still ``None``."""
        return tuple(name for name in self.REQUIRED if getattr(self, name) is None)

    def require(self) -> None:
        """Raise unless every required threshold has been supplied."""
        absent = self.missing()
        if absent:
            raise MissingThresholdsError(
                f"Diagnostic mode needs calibrated thresholds; {', '.join(absent)} "
                f"{'is' if len(absent) == 1 else 'are'} unset. There is deliberately no "
                f"default: an uncalibrated cut point driving selection is the defect this "
                f"replaces. See {_CALIBRATION_DOC}."
            )
        if self.concentration_lo is not None and self.concentration_hi is not None:
            if self.concentration_lo > self.concentration_hi:
                raise ValueError(
                    f"concentration_lo ({self.concentration_lo}) must not exceed "
                    f"concentration_hi ({self.concentration_hi})"
                )


@dataclass(frozen=True)
class Criterion:
    """One clause of a regime's conjunction, with the margin that decided it."""

    name: str
    satisfied: bool
    value: float
    threshold: float

    @property
    def margin(self) -> float:
        """Distance from the cut point. Small means the clause nearly flipped."""
        return abs(self.value - self.threshold)


@dataclass(frozen=True)
class Diagnosis:
    """The regime, what it recommends, and everything that produced it."""

    regime: AttentionRegime
    stats: SaliencyStats
    overall_acc: float
    hard_quantile_acc: float
    robustness_gap: float
    recommended: tuple[str, ...]
    confidence: float
    reason: str
    criteria: tuple[Criterion, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Flat record for the run record and the report artifact."""
        return {
            "regime": self.regime.value,
            "recommended": list(self.recommended),
            "confidence": self.confidence,
            "reason": self.reason,
            "overall_acc": self.overall_acc,
            "hard_quantile_acc": self.hard_quantile_acc,
            "robustness_gap": self.robustness_gap,
            "stats": self.stats.to_dict(),
            "criteria": [
                {
                    "name": c.name,
                    "satisfied": c.satisfied,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for c in self.criteria
            ],
        }


#: What each regime asks for. Named here rather than inline so the report and
#: the docs quote the same source as the code.
_RECOMMENDATION = {
    AttentionRegime.SHORTCUT_SUSPECTED: ("icd",),
    AttentionRegime.OBJECT_FOCUSED: ("aicd",),
    AttentionRegime.UNSTRUCTURED: ("church_noise",),
}


@dataclass(frozen=True)
class _Cuts:
    """Thresholds resolved to plain floats, after ``require()`` proved present."""

    concentration_lo: float
    concentration_hi: float
    border_mass_hi: float
    perturbation_shift_hi: float
    robustness_gap_hi: float

    @classmethod
    def of(cls, thresholds: DiagnosisThresholds) -> _Cuts:
        return cls(
            concentration_lo=float(thresholds.concentration_lo),  # type: ignore[arg-type]
            concentration_hi=float(thresholds.concentration_hi),  # type: ignore[arg-type]
            border_mass_hi=float(thresholds.border_mass_hi),  # type: ignore[arg-type]
            perturbation_shift_hi=float(thresholds.perturbation_shift_hi),  # type: ignore[arg-type]
            robustness_gap_hi=float(thresholds.robustness_gap_hi),  # type: ignore[arg-type]
        )


def _shortcut_criteria(
    stats: SaliencyStats, gap: float, cuts: _Cuts
) -> tuple[Criterion, ...]:
    """Diffuse, at the border, unstable, and a real gap."""
    concentration = stats.concentration
    border = stats.border_mass
    shift = _shift(stats)
    return (
        Criterion("diffuse", concentration < cuts.concentration_lo, concentration, cuts.concentration_lo),
        Criterion("border_heavy", border > cuts.border_mass_hi, border, cuts.border_mass_hi),
        Criterion("unstable", shift > cuts.perturbation_shift_hi, shift, cuts.perturbation_shift_hi),
        Criterion("robustness_gap", gap > cuts.robustness_gap_hi, gap, cuts.robustness_gap_hi),
    )


def _object_focused_criteria(
    stats: SaliencyStats, gap: float, cuts: _Cuts
) -> tuple[Criterion, ...]:
    """Concentrated, central, stable, and a real gap."""
    concentration = stats.concentration
    border = stats.border_mass
    shift = _shift(stats)
    return (
        Criterion("concentrated", concentration > cuts.concentration_hi, concentration, cuts.concentration_hi),
        Criterion("central", border <= cuts.border_mass_hi, border, cuts.border_mass_hi),
        Criterion("stable", shift <= cuts.perturbation_shift_hi, shift, cuts.perturbation_shift_hi),
        Criterion("robustness_gap", gap > cuts.robustness_gap_hi, gap, cuts.robustness_gap_hi),
    )


def _shift(stats: SaliencyStats) -> float:
    """``perturbation_shift``, or 0.0 when it was not measured.

    0.0 reads as "perfectly stable", which is the conservative direction: it
    pushes the rule towards OBJECT_FOCUSED (sharpen what is there) rather than
    towards SHORTCUT_SUSPECTED (replace it), and sharpening a model that did not
    need it is the cheaper mistake.
    """
    return 0.0 if stats.perturbation_shift is None else stats.perturbation_shift


def _fraction(criteria: tuple[Criterion, ...]) -> float:
    return sum(1 for c in criteria if c.satisfied) / len(criteria)


def _smallest_margin(criteria: tuple[Criterion, ...]) -> float:
    return min(c.margin for c in criteria)


def diagnose(
    stats: SaliencyStats,
    *,
    overall_acc: float,
    hard_quantile_acc: float,
    thresholds: DiagnosisThresholds,
) -> Diagnosis:
    """Classify the attention regime and say which intervention it indicates.

    ``stats`` is the batch-level :class:`~bnnr.analysis.saliency_stats.SaliencyStats`;
    ``overall_acc`` and ``hard_quantile_acc`` come from the evaluation result, so
    ``robustness_gap`` here is the same quantity the metrics carry.

    Raises :class:`MissingThresholdsError` when the thresholds are uncalibrated.

    How the regime is chosen, stated in full because an unexplainable rule is
    the thing this replaces:

    1. **The gap gates everything.** With ``robustness_gap`` at or below its
       threshold there is no robustness failure to act on, and the answer is
       UNSTRUCTURED regardless of what the maps look like. A directed
       intervention needs something to fix.
    2. **A map with no shape is not evidence.** With ``concentration`` inside
       the ``[lo, hi]`` band it is neither diffuse nor concentrated, so the
       maps cannot discriminate and the answer is UNSTRUCTURED.
    3. Otherwise both regimes' conjunctions are evaluated and the one with more
       satisfied clauses wins. A tie goes to the regime whose weakest clause has
       the larger margin, i.e. the one that is less nearly false.

    ``confidence`` is the fraction of the winning regime's four clauses that
    hold. It is a count, not a weighted score, and that is deliberate: a
    weighted scalar here would be another uncalibrated number of exactly the
    kind FIX-2-4 is removing. For UNSTRUCTURED it is the fraction of clauses
    that support *neither* intervention, so a confident "do nothing" and a
    borderline one are distinguishable.
    """
    thresholds.require()
    cuts = _Cuts.of(thresholds)
    gap = overall_acc - hard_quantile_acc

    shortcut = _shortcut_criteria(stats, gap, cuts)
    focused = _object_focused_criteria(stats, gap, cuts)

    gap_criterion = shortcut[-1]
    if not gap_criterion.satisfied:
        return _unstructured(
            stats, overall_acc, hard_quantile_acc, gap, shortcut, focused,
            reason="no robustness gap to act on",
        )

    if cuts.concentration_lo <= stats.concentration <= cuts.concentration_hi:
        return _unstructured(
            stats, overall_acc, hard_quantile_acc, gap, shortcut, focused,
            reason="saliency has no usable structure",
        )

    shortcut_score = _fraction(shortcut)
    focused_score = _fraction(focused)

    if shortcut_score > focused_score:
        winner, criteria, score = AttentionRegime.SHORTCUT_SUSPECTED, shortcut, shortcut_score
    elif focused_score > shortcut_score:
        winner, criteria, score = AttentionRegime.OBJECT_FOCUSED, focused, focused_score
    elif _smallest_margin(shortcut) >= _smallest_margin(focused):
        winner, criteria, score = AttentionRegime.SHORTCUT_SUSPECTED, shortcut, shortcut_score
    else:
        winner, criteria, score = AttentionRegime.OBJECT_FOCUSED, focused, focused_score

    satisfied = [c.name for c in criteria if c.satisfied]
    return Diagnosis(
        regime=winner,
        stats=stats,
        overall_acc=overall_acc,
        hard_quantile_acc=hard_quantile_acc,
        robustness_gap=gap,
        recommended=_RECOMMENDATION[winner],
        confidence=score,
        reason=f"{winner.value}: {', '.join(satisfied)}" if satisfied else winner.value,
        criteria=criteria,
    )


def _unstructured(
    stats: SaliencyStats,
    overall_acc: float,
    hard_quantile_acc: float,
    gap: float,
    shortcut: tuple[Criterion, ...],
    focused: tuple[Criterion, ...],
    *,
    reason: str,
) -> Diagnosis:
    """Neither intervention is indicated; say how firmly."""
    against = 1.0 - max(_fraction(shortcut), _fraction(focused))
    return Diagnosis(
        regime=AttentionRegime.UNSTRUCTURED,
        stats=stats,
        overall_acc=overall_acc,
        hard_quantile_acc=hard_quantile_acc,
        robustness_gap=gap,
        recommended=_RECOMMENDATION[AttentionRegime.UNSTRUCTURED],
        confidence=against,
        reason=f"unstructured: {reason}",
        criteria=(),
    )
