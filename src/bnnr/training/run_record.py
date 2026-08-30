"""What a run has to record about how it decided and what it cost.

Two findings drive this.

**The compute confound.** T20's apparent deficit under "equal compute" was the
epoch split, not the method: the deployed model trains for ``B/3`` while a
single-augmentation baseline trains the full ``B``. Matching deployed epochs
closed a 4.46 pp gap to 0.09 pp on Imagewoof. The old record carried
``max_iterations`` and ``m_epochs`` but neither of the two numbers that actually
separate the protocols, so the confound had to be reconstructed by hand months
later, from run directories, by someone who remembered it existed.

**The path.** Once behaviour depends on a selector, a diagnosis, or a search
policy, "BNNR vs baseline" is not a single number unless every row says which
path it took. A benchmark that mixes two selectors and reports one mean is
measuring their average, which is a quantity nobody wanted.

Both epoch counts are **counted, not derived**. Candidate pruning stops a
candidate early and the deployed model keeps its best epoch rather than its
last, so ``m_epochs * iterations`` is an upper bound rather than an answer.
:class:`ComputeLedger` counts what actually ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ComputeLedger",
    "RunRecord",
]


@dataclass
class ComputeLedger:
    """Counts epochs as they run, rather than inferring them from config.

    ``total_gpu_epochs`` is every epoch trained anywhere: the baseline phase,
    every candidate of every iteration, including candidates that were pruned
    and candidates that lost. It is what an equal-compute protocol has to match.

    ``deployed_epochs`` is how much training the model you ship actually
    received: the baseline's best epoch, plus the winning candidate's best
    epoch for each accepted iteration. It is what an equal-deployed-epoch
    protocol has to match. Best rather than last, because that is the checkpoint
    the run keeps.

    The two are equal only for a plain training run with no search.
    """

    total_gpu_epochs: int = 0
    deployed_epochs: int = 0

    def count_trained_epoch(self) -> None:
        """One epoch of training happened, for whatever candidate."""
        self.total_gpu_epochs += 1

    def credit_deployed(self, epochs: int) -> None:
        """*epochs* of the model that ships were kept, from a phase that ended."""
        if epochs > 0:
            self.deployed_epochs += epochs


@dataclass
class RunRecord:
    """The fields every run must carry for its result to be comparable.

    Anything optional here is optional because the feature that fills it has not
    landed yet, not because it is nice to have. ``search_policy`` is
    ``"exhaustive"`` for every run today; #413 introduces the alternatives and
    this field is what makes a benchmark row from before and after that change
    distinguishable.
    """

    #: Every epoch trained anywhere in the run.
    total_gpu_epochs: int = 0
    #: Epochs of training the shipped model received.
    deployed_epochs: int = 0

    #: How candidates were enumerated: exhaustive, diagnosis_single or
    #: successive_halving.
    search_policy: str = "exhaustive"
    #: The rung structure that policy produced, with its epoch accounting.
    #: ``None`` for a run that never reached the search phase.
    search_plan: dict[str, Any] | None = None
    #: Which rule picked the winner, from ``config.selector``.
    selector: str = "metric_argmax"
    #: Names the selector chose. A tuple, matching ``SelectionResult.selected``,
    #: because a policy that advances several candidates at once needs the
    #: plural and widening the type later would break every reader.
    selected_candidate: tuple[str, ...] = ()

    #: ``Diagnosis.to_dict()`` when one was computed, else ``None``. Shadow mode
    #: fills this on runs whose selector ignored it, which is the point of
    #: shadow mode.
    diagnosis: dict[str, Any] | None = None

    #: The loss quantile the robustness metrics used, from
    #: ``config.hard_quantile_q``. Recorded beside the numbers it produced so a
    #: reader never has to guess which sweep value a row belongs to.
    hard_quantile_q: float | None = None

    #: ``{augmentation_name: mode}`` for augmentations whose transform depends
    #: on a mode argument (``noise_mode``, ``effect_mode``, ``camera_mode``).
    #: Before those existed the device decided which transform ran, so a run
    #: without this is not reproducible across machines.
    augmentation_modes: dict[str, str] = field(default_factory=dict)

    #: Why the selector landed where it did: ``"improved"``,
    #: ``"no_improvement"``, ``"indistinguishable"``, ``"no_candidates"``.
    selection_reason: str | None = None
    #: The paired bootstrap interval behind that reason, when one was computed.
    #: Recorded so a decision is auditable after the fact rather than only at
    #: the moment it was made, which is what T20 had to reconstruct by hand.
    selection_interval: dict[str, float | int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping. ``selected_candidate`` becomes a list."""
        data = asdict(self)
        data["selected_candidate"] = list(self.selected_candidate)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RunRecord:
        """Rebuild from a record, tolerating one written before a field existed.

        Old records must keep summarizing rather than crashing a reader, so
        unknown keys are dropped and absent ones fall back to the field default.
        """
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        selected = kwargs.get("selected_candidate")
        # An explicit null is as good as absent: an older writer may have left
        # the key in place with no value.
        if selected is None:
            kwargs.pop("selected_candidate", None)
        else:
            kwargs["selected_candidate"] = tuple(selected)
        return cls(**kwargs)


#: Mode attributes an augmentation may carry, in the order they are looked up.
_MODE_ATTRS = ("noise_mode", "effect_mode", "camera_mode")


def collect_augmentation_modes(augmentations: list[Any]) -> dict[str, str]:
    """Read the mode of every augmentation that has one.

    Augmentations without a mode contribute nothing rather than a ``None``, so
    the recorded mapping stays a statement about what was configurable.
    """
    modes: dict[str, str] = {}
    for augmentation in augmentations:
        for attr in _MODE_ATTRS:
            mode = getattr(augmentation, attr, None)
            if mode is not None:
                modes[augmentation.name] = str(mode)
                break
    return modes
