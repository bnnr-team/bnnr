"""Candidate selection as a named, swappable component.

Selection used to be a branch inside ``select_best_path`` plus a composite score
with a weight nobody calibrated. That shape has two costs. The benchmark cannot
record *which* rule ran, so a contrast between two rules is not expressible; and
a new rule cannot compete with argmax on equal terms, because argmax is not a
thing you can name, it is the shape of the function.

This module makes the rule explicit. A selector takes the candidates, the
baseline and the config, and returns a :class:`SelectionResult` that says what
it picked, on what scores, and why. ``SELECTORS`` maps a config string to one.

Nothing here changes default behaviour. ``metric_argmax`` is the logic
``select_best_path`` has always run, moved rather than rewritten, and it stays
the default. ``select_best_path`` keeps its name, signature and ``str | None``
return, and is now a thin adapter over the registry.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bnnr.config_model import BNNRConfig

__all__ = [
    "SELECTORS",
    "CandidateReport",
    "CandidateSelector",
    "SelectionResult",
    "get_selector",
    "run_selector",
]


@dataclass(frozen=True)
class CandidateReport:
    """One candidate as a selector sees it.

    ``metrics`` is the candidate's evaluation result, so a selector reads the
    configured selection metric out of it rather than being handed a single
    pre-chosen number. That is what lets a selector look at something other
    than the metric being optimised, which is the entire point of the exercise.
    """

    name: str
    metrics: dict[str, float]
    xai_score: float | None = None

    def value(self, metric: str) -> float | None:
        """The named metric, or ``None`` when this candidate did not report it."""
        raw = self.metrics.get(metric)
        return None if raw is None else float(raw)


@dataclass(frozen=True)
class SelectionResult:
    """What a selector decided, and enough context to explain it afterwards.

    ``selected`` is a tuple rather than a single name. Today every selector
    returns zero or one, but a policy that advances several candidates at once
    (successive halving, FIX-4-1) needs the plural, and widening the type later
    would be the breaking change this avoids.

    ``reason`` is a short machine-readable slug, not prose: ``"improved"``,
    ``"no_improvement"``, ``"no_candidates"``. The run record stores it.
    """

    selected: tuple[str, ...]
    selector: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> str | None:
        """The single selected name, for callers that only handle one."""
        return self.selected[0] if self.selected else None


class CandidateSelector(Protocol):
    """Chooses which candidate augmentation to keep, if any."""

    name: str

    def select(
        self,
        candidates: list[CandidateReport],
        baseline: dict[str, float],
        config: BNNRConfig,
    ) -> SelectionResult:
        """Pick from *candidates*, or select nothing."""
        raise NotImplementedError


def _resolved_values(candidates: list[CandidateReport], metric: str) -> dict[str, float]:
    """Candidates that reported *metric*, in their original order."""
    resolved = {}
    for candidate in candidates:
        value = candidate.value(metric)
        if value is not None:
            resolved[candidate.name] = value
    return resolved


def _improved(value: float, baseline_value: float, mode: str) -> bool:
    return value > baseline_value if mode == "max" else value < baseline_value


def _gate_on_baseline(
    name: str,
    candidates: list[CandidateReport],
    baseline: dict[str, float],
    config: BNNRConfig,
    selector: str,
    scores: dict[str, float],
) -> SelectionResult:
    """Apply the shared rule: a pick only counts if it beat the baseline.

    Every selector goes through this, so a contrast between two selectors is a
    contrast between their ranking rules and nothing else. A selector that
    skipped the gate would look better simply by accepting runs the others
    rejected.
    """
    baseline_value = baseline.get(config.selection_metric)
    chosen = next((c for c in candidates if c.name == name), None)
    value = chosen.value(config.selection_metric) if chosen is not None else None

    if baseline_value is None or value is None:
        return SelectionResult((), selector, "no_baseline", scores)
    if not _improved(value, float(baseline_value), config.selection_mode):
        return SelectionResult((), selector, "no_improvement", scores)
    return SelectionResult((name,), selector, "improved", scores)


class MetricArgmaxSelector:
    """Greedy argmax on the selection metric, optionally blended with XAI quality.

    This is what BNNR has always done, moved here unchanged. Two paths, chosen
    by ``xai_selection_weight``:

    ``w <= 0`` or no XAI scores
        Plain argmax on the selection metric.
    ``w > 0``
        Composite of a min-max normalised metric and the XAI quality score.

    The composite path is the one T20 found wanting: the normalisation stretches
    whatever spread the candidates happened to have across the full [0, 1] range,
    so on a saturated metric the weight is applied to scatter rather than to
    signal. FIX-2-2 replaces the scale and FIX-2-4 deprecates the weight; both
    are deliberately out of scope here, because this change has to be provably
    behaviour-preserving before anything is allowed to alter what it does.
    """

    name = "metric_argmax"

    def select(
        self,
        candidates: list[CandidateReport],
        baseline: dict[str, float],
        config: BNNRConfig,
    ) -> SelectionResult:
        metric = config.selection_metric
        mode = config.selection_mode
        weight = config.xai_selection_weight

        values = _resolved_values(candidates, metric)
        if not values:
            return SelectionResult((), self.name, "no_candidates", {})

        has_xai = any(c.xai_score is not None for c in candidates)
        if weight <= 0 or not has_xai:
            sign = 1.0 if mode == "max" else -1.0
            best_name = max(values, key=lambda name: sign * values[name])
            return _gate_on_baseline(best_name, candidates, baseline, config, self.name, values)

        lo, hi = min(values.values()), max(values.values())
        span = hi - lo if hi != lo else 1.0
        xai_by_name = {c.name: (c.xai_score or 0.0) for c in candidates}

        scores = {}
        for name, value in values.items():
            normalised = (value - lo) / span if mode == "max" else (hi - value) / span
            scores[name] = (1.0 - weight) * normalised + weight * xai_by_name.get(name, 0.0)

        best_name = max(scores, key=lambda name: scores[name])
        return _gate_on_baseline(best_name, candidates, baseline, config, self.name, scores)


class RandomSelector:
    """Uniform pick among the candidates, still gated on beating the baseline.

    This is the arm T20 benchmarked ``selection_mode="xai"`` against and failed
    to beat, so it exists as a first-class selector rather than as something the
    benchmark harness improvises. Seeded from ``config.seed`` so a run is
    reproducible; the seed is mixed with the candidate names, otherwise every
    iteration of a run with the same arity would make the same choice.
    """

    name = "random"

    def select(
        self,
        candidates: list[CandidateReport],
        baseline: dict[str, float],
        config: BNNRConfig,
    ) -> SelectionResult:
        values = _resolved_values(candidates, config.selection_metric)
        if not values:
            return SelectionResult((), self.name, "no_candidates", {})

        # Not hash(): Python randomises str hashing per process, so a run would
        # not reproduce across invocations, which is the one thing the seed is for.
        digest = hashlib.sha256(
            "\x00".join([str(config.seed), *sorted(values)]).encode()
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        picked = rng.choice(sorted(values))
        scores = {name: (1.0 if name == picked else 0.0) for name in values}
        return _gate_on_baseline(picked, candidates, baseline, config, self.name, scores)


SELECTORS: dict[str, CandidateSelector] = {
    MetricArgmaxSelector.name: MetricArgmaxSelector(),
    RandomSelector.name: RandomSelector(),
}


def get_selector(name: str) -> CandidateSelector:
    """Look up a selector by its config name."""
    try:
        return SELECTORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown selector {name!r}. Available: {sorted(SELECTORS)}"
        ) from None


def run_selector(
    results: dict[str, dict[str, float]],
    baseline_metrics: dict[str, float],
    config: BNNRConfig,
    xai_scores: dict[str, float] | None = None,
) -> SelectionResult:
    """Build the candidate reports and run the configured selector over them."""
    candidates = [
        CandidateReport(
            name=name,
            metrics=metrics,
            # A non-empty dict whose keys miss this candidate still means "XAI
            # scoring is on", and the legacy code scored the miss as 0.0. None
            # is reserved for "no XAI scoring at all", which is the condition
            # that selects the plain-argmax path.
            xai_score=(xai_scores.get(name, 0.0) if xai_scores else None),
        )
        for name, metrics in results.items()
    ]
    return get_selector(config.selector).select(candidates, baseline_metrics, config)
