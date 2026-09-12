"""Augmentation candidate selection and pruning logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bnnr.training import selection as _selection

if TYPE_CHECKING:
    from bnnr.config_model import BNNRConfig


def select_best_path(
    results: dict[str, dict[str, float]],
    baseline_metrics: dict[str, float],
    config: BNNRConfig,
    xai_scores: dict[str, float] | None = None,
) -> str | None:
    """Pick the best augmentation candidate from *results*, or ``None`` if no improvement.

    Thin adapter over :mod:`bnnr.training.selection`. The logic that used to live
    here is now ``SELECTORS["metric_argmax"]``, which stays the default, so this
    returns exactly what it always did. Setting ``config.selector`` routes the
    same call through a different rule.

    Kept at this name and signature on purpose: it is public surface, imported
    by ``tests/test_backward_compat.py`` and re-exported from ``bnnr.training``.
    """
    return _selection.run_selector(results, baseline_metrics, config, xai_scores).best


def should_prune_candidate(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    config: BNNRConfig,
    xai_quality: float | None = None,
) -> bool:
    """Return ``True`` if the candidate should be pruned early."""
    if not config.candidate_pruning_enabled:
        return False
    metric = config.selection_metric
    candidate_value = candidate_metrics.get(metric)
    baseline_value = baseline_metrics.get(metric)
    if candidate_value is None or baseline_value is None:
        return False

    threshold = config.candidate_pruning_relative_threshold
    if config.selection_mode == "max":
        metric_prune = float(candidate_value) < float(baseline_value) * threshold
    else:
        metric_prune = float(candidate_value) > float(baseline_value) * (2.0 - threshold)

    if metric_prune:
        return True

    xai_thresh = config.xai_pruning_threshold
    if xai_thresh > 0 and xai_quality is not None and xai_quality < xai_thresh:
        return True

    return False


def get_current_best_metric(
    results: dict[str, dict[str, float]],
    config: BNNRConfig,
) -> float | None:
    """Return the best selection-metric value seen so far across candidates."""
    metric = config.selection_metric
    values = [v[metric] for v in results.values() if metric in v]
    if not values:
        return None
    return float(max(values) if config.selection_mode == "max" else min(values))


def top_k_candidate_names(
    results: dict[str, dict[str, float]],
    config: BNNRConfig,
    k: int = 3,
) -> list[str]:
    """Return names of up to *k* best candidates, ordered by selection metric."""
    metric = config.selection_metric
    sorted_items = sorted(
        ((name, metrics) for name, metrics in results.items() if metric in metrics),
        key=lambda item: float(item[1][metric]),
        reverse=(config.selection_mode == "max"),
    )
    return [name for name, _ in sorted_items[:k]]
