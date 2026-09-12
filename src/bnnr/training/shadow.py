"""Record the attention evidence on every run, without letting it decide anything.

The thresholds the diagnosis needs have no defaults and the library refuses
diagnostic mode until they are supplied. Something has to supply them, and the
only alternatives are to guess — the mistake this whole programme exists to
undo — or to collect evidence first.

Shadow mode collects it. Every run computes the saliency statistics and the
robustness metrics it was going to compute anyway, records them per candidate
alongside which candidate the run actually kept and how it turned out, and lets
none of it touch selection. From the moment it is on, a run that was going to
happen regardless becomes a sample of *(statistics, chosen arm, outcome)* at no
extra GPU cost.

This is why Phase 3 sits before Phase 4 rather than inside it.

**Records are raw statistics, never a regime.** Writing a regime would need
thresholds, and the thresholds are the thing being calibrated. A record that
already assumed an answer would be useless for finding it.

**Every candidate is recorded, not only the winner.** A calibration set with
only winning arms cannot answer "would the other choice have been better",
which is the question the whole exercise turns on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bnnr.analysis.saliency_stats import SaliencyStats

__all__ = [
    "SHADOW_RECORDS_FILENAME",
    "ShadowRecord",
    "ShadowRecorder",
]

#: One JSON object per line, in the run directory. JSONL rather than JSON so a
#: run killed halfway still leaves every record it had already written.
SHADOW_RECORDS_FILENAME = "shadow_records.jsonl"


@dataclass
class ShadowRecord:
    """One observation: what the attention looked like, and what happened.

    ``candidate`` is the augmentation name, or ``"baseline"`` for the phase
    before any augmentation was applied. ``selected`` says whether this arm is
    the one the run kept, which is what turns a pile of statistics into a
    supervised calibration set.
    """

    phase: str
    iteration: int
    candidate: str

    #: ``SaliencyStats.to_dict()``, or ``None`` when no maps were available.
    stats: dict[str, Any] | None = None

    overall_acc: float | None = None
    hard_quantile_acc: float | None = None
    robustness_gap: float | None = None

    #: How many images the statistics were computed over. Recorded because the
    #: probe set is a sample, and a statistic without its sample size cannot be
    #: weighted against another run's.
    sample_size: int = 0

    #: Whether this arm was the one the run kept. Filled in after selection, so
    #: it is False on every record until the iteration resolves.
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowRecorder:
    """Accumulates records for a run and writes them as JSONL.

    Held in memory until the run ends because ``selected`` is only knowable
    after the iteration resolves, and rewriting a line in place is worse than
    holding a few hundred small dicts.
    """

    records: list[ShadowRecord] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        iteration: int,
        candidate: str,
        stats: SaliencyStats | None,
        metrics: dict[str, float] | None,
    ) -> ShadowRecord:
        """Add one observation, reading the robustness fields off *metrics*.

        The metrics dict is whatever the evaluation returned. The three
        robustness keys are absent for multilabel and detection runs, and that
        is recorded as ``None`` rather than as zero: no measurement and a
        measurement of zero are different facts about a run.
        """
        metrics = metrics or {}
        entry = ShadowRecord(
            phase=phase,
            iteration=iteration,
            candidate=candidate,
            stats=stats.to_dict() if stats is not None else None,
            overall_acc=_maybe_float(metrics.get("accuracy")),
            hard_quantile_acc=_maybe_float(metrics.get("hard_quantile_acc")),
            robustness_gap=_maybe_float(metrics.get("robustness_gap")),
            sample_size=stats.n_maps if stats is not None else 0,
        )
        self.records.append(entry)
        return entry

    def mark_selected(self, iteration: int, candidate: str | None) -> None:
        """Flag the arm this iteration kept, if it kept one."""
        if candidate is None:
            return
        for entry in self.records:
            if entry.iteration == iteration and entry.candidate == candidate:
                entry.selected = True

    def write(self, run_dir: Path) -> Path | None:
        """Write the records to *run_dir*, or do nothing when there are none."""
        if not self.records:
            return None
        path = run_dir / SHADOW_RECORDS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(entry.to_dict()) for entry in self.records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _maybe_float(value: Any) -> float | None:
    """Coerce to float, keeping the difference between absent and zero."""
    return None if value is None else float(value)


def stats_from_maps(maps: Any, *, n_maps: int | None = None) -> SaliencyStats | None:
    """Aggregate saliency statistics over a batch of maps.

    Returns ``None`` rather than a zeroed record when there are no maps, so a
    run with XAI disabled contributes nothing instead of contributing a
    misleading "perfectly uniform attention" sample.

    ``perturbation_shift`` is deliberately not computed here. It needs a second
    explainer pass, and shadow mode's entire claim is that it costs nothing;
    paying for it silently on every run would break that. The three pure
    statistics come free from maps the run already produced.
    """
    import numpy as np

    from bnnr.analysis.saliency_stats import aggregate_saliency_stats, saliency_stats_from_map

    array = np.asarray(maps)
    if array.size == 0:
        return None
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        return None

    per_map = [saliency_stats_from_map(array[i]) for i in range(array.shape[0])]
    if not per_map:
        return None
    aggregate = aggregate_saliency_stats(per_map)
    if n_maps is not None and n_maps != aggregate.n_maps:
        from dataclasses import replace

        aggregate = replace(aggregate, n_maps=n_maps)
    return aggregate
