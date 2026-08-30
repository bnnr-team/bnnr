"""How the candidate budget is spent, as a named and swappable plan.

This is the headline recommendation of the T20 findings: one design change
closes both defects the benchmarks identified.

**The selection criterion.** ``exhaustive`` evaluates every candidate and then
arbitrates on selection-validation accuracy, a quantity T20 found close to
orthogonal to the objective. ``diagnosis_single`` skips the arbitration: the
attention diagnosis names one candidate and that candidate trains.

**The epoch split.** Under ``exhaustive`` the deployed model receives roughly
``B/3`` of the budget while a single-augmentation baseline receives all of ``B``,
which is what made BNNR look worse than it is. Matching deployed epochs closed
4.46 pp to 0.09 pp on Imagewoof. ``diagnosis_single`` gives its one candidate
the whole budget; ``successive_halving`` kills weak branches after a rung
instead of paying a flat ``B/3`` for each.

A policy produces a :class:`SearchPlan` of rungs. The loop executes rungs; it
does not decide what they contain. That separation is what lets a policy be
swapped, recorded, and benchmarked against another one.

**Defaults do not move here.** ``exhaustive`` stays the default until the
calibration study says otherwise, and it produces exactly the plan the loop
used to hard-code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bnnr.analysis.diagnosis import Diagnosis
    from bnnr.config_model import BNNRConfig

__all__ = [
    "DIAGNOSIS_DRIVEN_POLICIES",
    "SEARCH_POLICIES",
    "SearchPlan",
    "SearchRung",
    "UnplannableSearchError",
    "plan_search",
]


class UnplannableSearchError(RuntimeError):
    """A policy was asked for a plan it cannot produce."""


@dataclass(frozen=True)
class SearchRung:
    """One round of training: these candidates, this many epochs each."""

    candidates: tuple[str, ...]
    epochs: int
    #: How many of these survive into the next rung. Equal to ``len(candidates)``
    #: on a final rung, where survival means "is eligible to be selected".
    survivors: int

    @property
    def cost(self) -> int:
        """Epochs this rung spends in total."""
        return len(self.candidates) * self.epochs


@dataclass(frozen=True)
class SearchPlan:
    """What a policy decided to spend the iteration's budget on."""

    policy: str
    rungs: tuple[SearchRung, ...]

    @property
    def total_epochs(self) -> int:
        """Every epoch this plan will train, across all rungs."""
        return sum(rung.cost for rung in self.rungs)

    @property
    def deployed_epochs(self) -> int:
        """Epochs the surviving candidate accumulates, if it survives every rung.

        This is the number an equal-deployed-epoch protocol matches, and the
        one ``exhaustive`` starves.
        """
        return sum(rung.epochs for rung in self.rungs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "total_epochs": self.total_epochs,
            "deployed_epochs": self.deployed_epochs,
            "rungs": [
                {
                    "candidates": list(rung.candidates),
                    "epochs": rung.epochs,
                    "survivors": rung.survivors,
                }
                for rung in self.rungs
            ],
        }


def _exhaustive(
    candidates: tuple[str, ...], config: BNNRConfig, diagnosis: Diagnosis | None
) -> SearchPlan:
    """Every candidate, ``m_epochs`` each, one rung. Today's behaviour exactly."""
    del diagnosis
    return SearchPlan(
        "exhaustive",
        (SearchRung(candidates, config.m_epochs, survivors=len(candidates)),),
    )


def _diagnosis_single(
    candidates: tuple[str, ...], config: BNNRConfig, diagnosis: Diagnosis | None
) -> SearchPlan:
    """One candidate, named by the diagnosis, trained for the whole budget.

    The budget is ``m_epochs * len(candidates)``: exactly what ``exhaustive``
    would have spent on this iteration, now spent on the arm the evidence
    points at instead of split across arms that get thrown away.

    Refuses rather than guessing when there is no diagnosis, or when the
    diagnosis recommends nothing among the candidates. Falling back to argmax
    silently would make a benchmark contrast between the two policies measure a
    blend of them, which is the same reason the diagnosis selector refuses.
    """
    if diagnosis is None:
        raise UnplannableSearchError(
            "search_policy='diagnosis_single' needs a diagnosis for the iteration. "
            "Supply calibrated thresholds so one can be computed; see docs/diagnosis.md."
        )
    wanted = _match_recommended(candidates, diagnosis.recommended)
    if wanted is None:
        raise UnplannableSearchError(
            f"The diagnosis recommends {list(diagnosis.recommended)}, which matches none "
            f"of the candidates {list(candidates)}. Add a candidate of that family, or "
            f"use search_policy='exhaustive'."
        )
    budget = config.m_epochs * max(len(candidates), 1)
    return SearchPlan("diagnosis_single", (SearchRung((wanted,), budget, survivors=1),))


def _successive_halving(
    candidates: tuple[str, ...], config: BNNRConfig, diagnosis: Diagnosis | None
) -> SearchPlan:
    """Halve the field each rung, spending the survivors' budget on them.

    The fallback when the diagnosis is uncertain. Weak branches die after a
    short rung instead of each taking a flat ``m_epochs``, so the epochs they
    would have burned go to the arms still in contention.

    Rung epochs are chosen so the plan costs no more than ``exhaustive``: the
    first rung is short, and what the eliminated candidates would have spent is
    handed to the survivors. A single candidate degenerates to one full-budget
    rung, which is the right answer rather than a special case.
    """
    del diagnosis
    n = len(candidates)
    if n <= 1:
        return SearchPlan(
            "successive_halving",
            (SearchRung(candidates, config.m_epochs * max(n, 1), survivors=n),),
        )

    budget = config.m_epochs * n
    n_rungs = max(1, int(math.floor(math.log2(n))) + 1)
    # Split the budget evenly across rungs, so each rung's survivors get a
    # comparable slice of training rather than the last rung taking everything.
    per_rung = max(1, budget // (n_rungs * n))

    rungs: list[SearchRung] = []
    alive = candidates
    for index in range(n_rungs):
        last = index == n_rungs - 1
        survivors = len(alive) if last else max(1, len(alive) // 2)
        # Survivors of a shrinking field can afford longer rungs; give the
        # remaining budget to the final one rather than leaving it unspent.
        epochs = per_rung if not last else max(1, config.m_epochs - per_rung * index)
        rungs.append(SearchRung(alive, epochs, survivors=survivors))
        if last:
            break
        alive = alive[:survivors]
    return SearchPlan("successive_halving", tuple(rungs))


#: Longest first, so a family that contains another as a substring is tested
#: before the shorter one can claim its candidates. Mirrors the selector.
_FAMILY_ALIASES = ("aicd", "icd", "church_noise")


def _match_recommended(
    candidates: tuple[str, ...], recommended: tuple[str, ...]
) -> str | None:
    """First candidate belonging to a recommended family, or ``None``."""
    for family in recommended:
        for name in candidates:
            lowered = name.lower()
            matched = next((a for a in _FAMILY_ALIASES if a in lowered), None)
            if (matched == family) if matched else (family in lowered):
                return name
    return None


SEARCH_POLICIES = {
    "exhaustive": _exhaustive,
    "diagnosis_single": _diagnosis_single,
    "successive_halving": _successive_halving,
}

#: Policies that cannot run without a diagnosis, and therefore without
#: calibrated thresholds. The config validator reads this.
DIAGNOSIS_DRIVEN_POLICIES = frozenset({"diagnosis_single"})


def plan_search(
    candidates: tuple[str, ...],
    config: BNNRConfig,
    *,
    diagnosis: Diagnosis | None = None,
) -> SearchPlan:
    """Build the plan for one iteration under the configured policy."""
    try:
        builder = SEARCH_POLICIES[config.search_policy]
    except KeyError:
        raise ValueError(
            f"Unknown search_policy {config.search_policy!r}. "
            f"Available: {sorted(SEARCH_POLICIES)}"
        ) from None
    return builder(candidates, config, diagnosis)
