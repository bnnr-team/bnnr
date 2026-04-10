"""Regression: heterogeneous metric dicts from skipped detection batches."""

from __future__ import annotations

from bnnr.core import BNNRTrainer


def test_average_metrics_union_keys_skips_missing() -> None:
    """When some batches omit loss components (skipped non-finite batch), averaging must not KeyError."""
    trainer = BNNRTrainer.__new__(BNNRTrainer)
    out = BNNRTrainer._average_metrics(
        trainer,
        [
            {"loss": 2.0, "loss_loss_classifier": 0.5, "loss_loss_box_reg": 1.5},
            {"loss": 0.0, "loss_non_finite": 1.0},
        ],
    )
    assert out["loss"] == 1.0
    assert out["loss_non_finite"] == 1.0
    assert out["loss_loss_classifier"] == 0.5
    assert out["loss_loss_box_reg"] == 1.5


def test_average_metrics_ignores_non_finite() -> None:
    trainer = BNNRTrainer.__new__(BNNRTrainer)
    out = BNNRTrainer._average_metrics(
        trainer,
        [
            {"loss": 1.0},
            {"loss": float("nan")},
            {"loss": 3.0},
        ],
    )
    assert out["loss"] == 2.0
