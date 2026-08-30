"""Is the winning candidate actually distinguishable from the baseline?

T20's central negative result is that the selector picks between candidates
that are not distinguishable on the criterion it uses. The method had no way to
say so: ``select_best_path`` always returned a winner whenever its metric was
larger by any margin at all, and the run record made that look like a decision.

On Waterbirds the candidate accuracies were .8749 / .8816 / .8549 on a
validation set where the binomial standard error is larger than the spread. A
strict ``>`` on those numbers is a coin flip wearing a lab coat.

This module answers the question properly: bootstrap the *paired* difference in
per-sample correctness between two arms, and if the confidence interval covers
zero, say so rather than picking.

**Paired, not two-sample.** Both arms are evaluated on the same validation set,
so the samples are matched and the pairing removes the variance that comes from
the set itself rather than from the arms. Ignoring it inflates the interval and
would make everything look indistinguishable.

**Mean, not median.** The seed-level convention in the benchmark summarizers is
a median paired difference, because a per-seed metric is continuous and skewed.
Here the paired difference per sample is in ``{-1, 0, 1}``, where the median is
degenerate: it is 0 unless one arm is right on more than half the samples the
other got wrong. The mean of those differences *is* the accuracy difference,
which is the quantity being tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "PairedInterval",
    "paired_bootstrap_ci",
]

#: Resamples for the bootstrap. Enough for a stable 95% interval on a
#: validation set, cheap enough to run on every iteration: the resampling is
#: over a boolean vector, not over the model.
DEFAULT_RESAMPLES = 2000

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PairedInterval:
    """A bootstrap interval on the paired difference between two arms."""

    #: Observed mean paired difference, ``a - b``. Positive means *a* is better.
    difference: float
    low: float
    high: float
    confidence: float
    n_pairs: int
    n_resamples: int

    @property
    def contains_zero(self) -> bool:
        """Whether the interval covers zero, i.e. the arms are indistinguishable."""
        return self.low <= 0.0 <= self.high

    def to_dict(self) -> dict[str, float | int]:
        """JSON-ready, for the run record."""
        return {
            "difference": self.difference,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "n_pairs": self.n_pairs,
            "n_resamples": self.n_resamples,
        }


def paired_bootstrap_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> PairedInterval | None:
    """Percentile bootstrap interval on the mean paired difference ``a - b``.

    ``correct_a`` and ``correct_b`` are per-sample correctness for the same
    validation samples in the same order. Pass boolean or 0/1 arrays.

    Returns ``None`` when the inputs cannot support an interval: different
    lengths, or fewer than two pairs. ``None`` means "no test was possible",
    which callers must not confuse with "the arms tie" — the first should fall
    back to the old comparison, the second should refuse to switch.

    Resampling is over the *pairs*, indexing both arms with the same draw,
    which is what makes the interval paired.
    """
    a = np.asarray(correct_a).astype(np.float64).ravel()
    b = np.asarray(correct_b).astype(np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return None

    differences = a - b
    observed = float(differences.mean())

    rng = np.random.default_rng(seed)
    # One draw of indices per resample, applied to the difference vector, so
    # both arms are resampled together.
    idx = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    means = differences[idx].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return PairedInterval(
        difference=observed,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_pairs=int(differences.size),
        n_resamples=int(n_resamples),
    )


def correctness_vector(preds: np.ndarray | None, labels: np.ndarray | None) -> np.ndarray | None:
    """Per-sample correctness from cached predictions, or ``None``.

    Returns ``None`` rather than an empty array when either side is missing, so
    a caller cannot accidentally run a test on nothing.
    """
    if preds is None or labels is None:
        return None
    preds_arr = np.asarray(preds)
    labels_arr = np.asarray(labels)
    if preds_arr.shape != labels_arr.shape or preds_arr.size == 0:
        return None
    return np.asarray(preds_arr == labels_arr)
