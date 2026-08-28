"""Accuracy on the hardest validation samples, as a label-free robustness proxy.

The decision rule in the attention diagnosis needs to know whether a model is
poorly robust to context shift. Group labels would say so directly, but
consuming them forfeits the assumption-class advantage that is the whole point
of BNNR: it asks for images and labels and nothing else. The JTT/EIIL family
solves this by inferring the hard group from the loss, and that is what this
does.

``hard_quantile_acc`` is accuracy restricted to the highest-loss ``q`` fraction
of the validation set. ``robustness_gap`` is ``overall_acc - hard_quantile_acc``.
A model that is uniformly mediocre has a small gap; a model that is excellent on
the majority and fails a minority has a large one, which is the shape of a
shortcut.

``q`` is a knob, not a constant. It defaults to 0.2, travels in the evaluation
result next to the two numbers it produced, and is one of the parameters the
threshold calibration study sweeps.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HARD_QUANTILE_KEYS",
    "hard_quantile_metrics",
]

#: Keys this module contributes to an evaluation result, so callers that filter
#: metrics do not have to spell them out one at a time.
HARD_QUANTILE_KEYS = ("hard_quantile_acc", "robustness_gap", "hard_quantile_q")


def hard_quantile_metrics(
    losses: np.ndarray,
    correct: np.ndarray,
    *,
    q: float,
) -> dict[str, float]:
    """Accuracy on the highest-loss ``q`` fraction, and the gap to overall.

    Parameters
    ----------
    losses
        Per-sample loss, shape ``(N,)``. Higher is harder.
    correct
        Whether each sample was predicted correctly, shape ``(N,)``, bool or 0/1.
    q
        Fraction of the set to treat as hard, in ``(0, 1]``.

    Returns a dict with ``hard_quantile_acc``, ``robustness_gap`` and the ``q``
    that produced them. An empty input returns an empty dict rather than a
    fabricated zero: no samples means no measurement, and a 0.0 here would read
    as a perfectly robust model.

    The count is rounded up, so a small validation set still gets at least one
    hard sample. Samples that tie on loss at the boundary are ordered by index,
    which makes the selection deterministic without claiming the tie was broken
    on anything meaningful.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")

    loss_arr = np.asarray(losses, dtype=np.float64).ravel()
    correct_arr = np.asarray(correct).astype(bool).ravel()
    if loss_arr.size != correct_arr.size:
        raise ValueError(
            f"losses and correct must have the same length, got {loss_arr.size} and {correct_arr.size}"
        )
    if loss_arr.size == 0:
        return {}

    n = loss_arr.size
    k = max(1, int(np.ceil(q * n)))
    # Stable sort so the boundary is broken by index rather than by whatever
    # order a partition happened to produce.
    hardest = np.argsort(-loss_arr, kind="stable")[:k]

    overall_acc = float(correct_arr.mean())
    hard_acc = float(correct_arr[hardest].mean())
    return {
        "hard_quantile_acc": hard_acc,
        "robustness_gap": overall_acc - hard_acc,
        "hard_quantile_q": float(q),
    }
