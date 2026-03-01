"""Tests for predefined + callable custom metrics.

Covers:
- Existing predefined metrics (cohen_kappa, mcc, balanced_accuracy)
- fbeta_score with various beta values (single-label and multi-label)
- f1_weighted, precision/recall variants (macro/micro/weighted)
- jaccard_score variants (samples/macro/micro/weighted)
- hamming (single-label), zero_one_loss
- Unsupported metric raises ValueError
- Custom callable metrics via BNNRTrainer
- validate_config acceptance of all new metrics
- Events system forwarding of extra metrics
"""
from __future__ import annotations

import numpy as np
import pytest

from bnnr.utils import _parse_fbeta, calculate_metrics

# ═══════════════════════════════════════════════════════════════════════
#  Helper fixtures for multi-label data
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def ml_data() -> tuple[np.ndarray, np.ndarray]:
    """Multi-label (2D) predictions and labels for 4 samples × 3 labels."""
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    y_pred = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
    ])
    return y_pred, y_true


@pytest.fixture
def sl_data() -> tuple[np.ndarray, np.ndarray]:
    """Single-label predictions and labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    return y_pred, y_true


# ═══════════════════════════════════════════════════════════════════════
#  _parse_fbeta helper
# ═══════════════════════════════════════════════════════════════════════

def test_parse_fbeta_valid() -> None:
    assert _parse_fbeta("fbeta_0.5") == pytest.approx(0.5)
    assert _parse_fbeta("fbeta_2") == pytest.approx(2.0)
    assert _parse_fbeta("fbeta_1") == pytest.approx(1.0)
    assert _parse_fbeta("fbeta_0.1") == pytest.approx(0.1)


def test_parse_fbeta_invalid() -> None:
    assert _parse_fbeta("f1_macro") is None
    assert _parse_fbeta("accuracy") is None
    assert _parse_fbeta("fbeta_abc") is None
    assert _parse_fbeta("fbeta_") is None


# ═══════════════════════════════════════════════════════════════════════
#  Existing predefined metrics (single-label)
# ═══════════════════════════════════════════════════════════════════════

def test_cohen_kappa_known_values() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])  # perfect
    result = calculate_metrics(y_pred, y_true, metrics=["cohen_kappa"])
    assert result["cohen_kappa"] == pytest.approx(1.0)


def test_mcc_known_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # perfect binary
    result = calculate_metrics(y_pred, y_true, metrics=["mcc"])
    assert result["mcc"] == pytest.approx(1.0)


def test_balanced_accuracy_known_values() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])  # perfect
    result = calculate_metrics(y_pred, y_true, metrics=["balanced_accuracy"])
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_balanced_accuracy_imbalanced() -> None:
    # All predicted as class 0 — regular accuracy would be 0.5 for a balanced set,
    # but for a 4:2 split it's 4/6. balanced_accuracy should be 0.5 (= mean of recalls).
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    result = calculate_metrics(y_pred, y_true, metrics=["balanced_accuracy"])
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_all_new_metrics_together() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    result = calculate_metrics(
        y_pred, y_true,
        metrics=["accuracy", "cohen_kappa", "mcc", "balanced_accuracy"],
    )
    assert "accuracy" in result
    assert "cohen_kappa" in result
    assert "mcc" in result
    assert "balanced_accuracy" in result
    # All values must be in [-1, 1] or [0, 1]
    assert -1.0 <= result["cohen_kappa"] <= 1.0
    assert -1.0 <= result["mcc"] <= 1.0
    assert 0.0 <= result["balanced_accuracy"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  fbeta_score — single-label
# ═══════════════════════════════════════════════════════════════════════

def test_fbeta_0_5_single_label_perfect() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_0.5"])
    assert result["fbeta_0.5"] == pytest.approx(1.0)


def test_fbeta_2_single_label_perfect() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_2"])
    assert result["fbeta_2"] == pytest.approx(1.0)


def test_fbeta_0_5_single_label_imperfect(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_0.5"])
    assert "fbeta_0.5" in result
    assert 0.0 <= result["fbeta_0.5"] <= 1.0


def test_fbeta_2_single_label_imperfect(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_2"])
    assert "fbeta_2" in result
    assert 0.0 <= result["fbeta_2"] <= 1.0


def test_fbeta_multiple_betas(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Multiple fbeta metrics with different betas computed together."""
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_0.5", "fbeta_1", "fbeta_2"])
    assert "fbeta_0.5" in result
    assert "fbeta_1" in result
    assert "fbeta_2" in result
    # fbeta_1 should equal f1_macro (macro average with beta=1)
    f1_result = calculate_metrics(y_pred, y_true, metrics=["f1_macro"])
    assert result["fbeta_1"] == pytest.approx(f1_result["f1_macro"])


# ═══════════════════════════════════════════════════════════════════════
#  fbeta_score — multi-label
# ═══════════════════════════════════════════════════════════════════════

def test_fbeta_0_5_multilabel_perfect() -> None:
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0]])
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_0.5"])
    assert result["fbeta_0.5"] == pytest.approx(1.0)


def test_fbeta_2_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_2"])
    assert "fbeta_2" in result
    assert 0.0 <= result["fbeta_2"] <= 1.0


def test_fbeta_0_5_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["fbeta_0.5"])
    assert "fbeta_0.5" in result
    assert 0.0 <= result["fbeta_0.5"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  f1_weighted — single-label and multi-label
# ═══════════════════════════════════════════════════════════════════════

def test_f1_weighted_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["f1_weighted"])
    assert "f1_weighted" in result
    assert 0.0 <= result["f1_weighted"] <= 1.0


def test_f1_weighted_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["f1_weighted"])
    assert "f1_weighted" in result
    assert 0.0 <= result["f1_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  precision variants — single-label
# ═══════════════════════════════════════════════════════════════════════

def test_precision_macro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_macro"])
    assert 0.0 <= result["precision_macro"] <= 1.0


def test_precision_micro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_micro"])
    assert 0.0 <= result["precision_micro"] <= 1.0


def test_precision_weighted_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_weighted"])
    assert 0.0 <= result["precision_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  precision variants — multi-label
# ═══════════════════════════════════════════════════════════════════════

def test_precision_macro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_macro"])
    assert 0.0 <= result["precision_macro"] <= 1.0


def test_precision_micro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_micro"])
    assert 0.0 <= result["precision_micro"] <= 1.0


def test_precision_weighted_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["precision_weighted"])
    assert 0.0 <= result["precision_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  recall variants — single-label
# ═══════════════════════════════════════════════════════════════════════

def test_recall_macro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_macro"])
    assert 0.0 <= result["recall_macro"] <= 1.0


def test_recall_micro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_micro"])
    assert 0.0 <= result["recall_micro"] <= 1.0


def test_recall_weighted_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_weighted"])
    assert 0.0 <= result["recall_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  recall variants — multi-label
# ═══════════════════════════════════════════════════════════════════════

def test_recall_macro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_macro"])
    assert 0.0 <= result["recall_macro"] <= 1.0


def test_recall_micro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_micro"])
    assert 0.0 <= result["recall_micro"] <= 1.0


def test_recall_weighted_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["recall_weighted"])
    assert 0.0 <= result["recall_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  jaccard_score — single-label and multi-label
# ═══════════════════════════════════════════════════════════════════════

def test_jaccard_macro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_macro"])
    assert 0.0 <= result["jaccard_macro"] <= 1.0


def test_jaccard_micro_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_micro"])
    assert 0.0 <= result["jaccard_micro"] <= 1.0


def test_jaccard_weighted_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_weighted"])
    assert 0.0 <= result["jaccard_weighted"] <= 1.0


def test_jaccard_samples_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_samples"])
    assert 0.0 <= result["jaccard_samples"] <= 1.0


def test_jaccard_macro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_macro"])
    assert 0.0 <= result["jaccard_macro"] <= 1.0


def test_jaccard_micro_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_micro"])
    assert 0.0 <= result["jaccard_micro"] <= 1.0


def test_jaccard_weighted_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["jaccard_weighted"])
    assert 0.0 <= result["jaccard_weighted"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  hamming (single-label) + zero_one_loss
# ═══════════════════════════════════════════════════════════════════════

def test_hamming_single_label_perfect() -> None:
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    result = calculate_metrics(y_pred, y_true, metrics=["hamming"])
    assert result["hamming"] == pytest.approx(1.0)


def test_hamming_single_label_imperfect(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["hamming"])
    assert 0.0 <= result["hamming"] <= 1.0


def test_zero_one_loss_single_label(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = sl_data
    result = calculate_metrics(y_pred, y_true, metrics=["zero_one_loss"])
    assert 0.0 <= result["zero_one_loss"] <= 1.0
    # zero_one_loss should be 1 - accuracy
    acc = calculate_metrics(y_pred, y_true, metrics=["accuracy"])
    assert result["zero_one_loss"] == pytest.approx(1.0 - acc["accuracy"])


def test_zero_one_loss_multilabel(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    y_pred, y_true = ml_data
    result = calculate_metrics(y_pred, y_true, metrics=["zero_one_loss"])
    assert 0.0 <= result["zero_one_loss"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  All metrics combined in a single call
# ═══════════════════════════════════════════════════════════════════════

def test_all_single_label_metrics_together(sl_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Verify ALL single-label metrics can be computed in a single call."""
    y_pred, y_true = sl_data
    all_metrics = [
        "accuracy", "f1_macro", "f1_micro", "f1_weighted",
        "precision", "precision_macro", "precision_micro", "precision_weighted",
        "recall", "recall_macro", "recall_micro", "recall_weighted",
        "cohen_kappa", "mcc", "balanced_accuracy",
        "hamming", "jaccard_macro", "jaccard_micro", "jaccard_weighted",
        "zero_one_loss", "fbeta_0.5", "fbeta_1", "fbeta_2",
    ]
    result = calculate_metrics(y_pred, y_true, metrics=all_metrics)
    assert len(result) == len(all_metrics)
    for m in all_metrics:
        assert m in result, f"Missing metric: {m}"


def test_all_multilabel_metrics_together(ml_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Verify ALL multi-label metrics can be computed in a single call."""
    y_pred, y_true = ml_data
    all_metrics = [
        "accuracy", "f1_samples", "f1_macro", "f1_micro", "f1_weighted",
        "precision", "precision_macro", "precision_micro", "precision_weighted",
        "recall", "recall_macro", "recall_micro", "recall_weighted",
        "hamming", "jaccard_samples", "jaccard_macro", "jaccard_micro", "jaccard_weighted",
        "zero_one_loss", "fbeta_0.5", "fbeta_1", "fbeta_2",
    ]
    result = calculate_metrics(y_pred, y_true, metrics=all_metrics)
    assert len(result) == len(all_metrics)
    for m in all_metrics:
        assert m in result, f"Missing metric: {m}"


# ═══════════════════════════════════════════════════════════════════════
#  validate_config acceptance of new metrics
# ═══════════════════════════════════════════════════════════════════════

def test_validate_config_accepts_fbeta_classification(tmp_path: object) -> None:
    from bnnr.config import validate_config
    from bnnr.core import BNNRConfig

    cfg = BNNRConfig(
        m_epochs=1, max_iterations=1,
        metrics=["accuracy", "fbeta_0.5", "loss"],
        selection_metric="fbeta_0.5",
        selection_mode="max",
        device="cpu",
        xai_enabled=False,
    )
    warnings = validate_config(cfg)
    assert not any("selection_metric" in w for w in warnings), (
        f"Unexpected warning about selection_metric: {warnings}"
    )


def test_validate_config_accepts_fbeta_multilabel(tmp_path: object) -> None:
    from bnnr.config import validate_config
    from bnnr.core import BNNRConfig

    cfg = BNNRConfig(
        task="multilabel",
        m_epochs=1, max_iterations=1,
        metrics=["fbeta_2", "f1_samples", "loss"],
        selection_metric="fbeta_2",
        selection_mode="max",
        device="cpu",
        xai_enabled=False,
    )
    warnings = validate_config(cfg)
    assert not any("selection_metric" in w for w in warnings), (
        f"Unexpected warning about selection_metric: {warnings}"
    )


def test_validate_config_accepts_jaccard(tmp_path: object) -> None:
    from bnnr.config import validate_config
    from bnnr.core import BNNRConfig

    cfg = BNNRConfig(
        m_epochs=1, max_iterations=1,
        metrics=["accuracy", "jaccard_macro", "loss"],
        selection_metric="jaccard_macro",
        selection_mode="max",
        device="cpu",
        xai_enabled=False,
    )
    warnings = validate_config(cfg)
    assert not any("selection_metric" in w for w in warnings)


def test_validate_config_rejects_invalid_metric() -> None:
    from bnnr.config import validate_config
    from bnnr.core import BNNRConfig

    cfg = BNNRConfig(
        m_epochs=1, max_iterations=1,
        metrics=["accuracy", "totally_bogus", "loss"],
        selection_metric="totally_bogus",
        selection_mode="max",
        device="cpu",
        xai_enabled=False,
    )
    warnings = validate_config(cfg)
    assert any("selection_metric" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════
#  Events system forwards extra metrics
# ═══════════════════════════════════════════════════════════════════════

def test_events_forward_extra_metrics() -> None:
    """epoch_end events with extra metrics (like fbeta_0.5) should be
    forwarded into the metrics_timeline row."""
    from bnnr.events import IncrementalReplayState

    acc = IncrementalReplayState()
    events = [
        {
            "type": "epoch_end",
            "payload": {
                "iteration": 0,
                "epoch": 1,
                "branch": "baseline",
                "metrics": {
                    "loss": 0.5,
                    "accuracy": 0.8,
                    "f1_macro": 0.75,
                    "fbeta_0.5": 0.82,
                    "jaccard_macro": 0.7,
                },
            },
        },
    ]
    acc.apply_events(events)
    row = acc.metrics_timeline[0]
    assert row["accuracy"] == pytest.approx(0.8)
    assert row["f1_macro"] == pytest.approx(0.75)
    # Extra metrics must also be present
    assert row["fbeta_0.5"] == pytest.approx(0.82)
    assert row["jaccard_macro"] == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════
#  Error handling
# ═══════════════════════════════════════════════════════════════════════

def test_unsupported_metric_raises() -> None:
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="Unsupported metric"):
        calculate_metrics(y_pred, y_true, metrics=["nonexistent_metric"])


def test_unsupported_metric_raises_multilabel() -> None:
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Unsupported metric"):
        calculate_metrics(y_pred, y_true, metrics=["nonexistent_metric"])


# ── Custom callable support on BNNRTrainer ───────────────────────────

def test_custom_metric_callable_receives_correct_shapes(
    model_adapter,
    dummy_dataloader,
    sample_config,
) -> None:
    """Ensure custom metric callables are called with correct arrays."""
    from bnnr.augmentations import BasicAugmentation
    from bnnr.core import BNNRTrainer

    call_log: list[tuple[np.ndarray, np.ndarray]] = []

    def _my_metric(preds: np.ndarray, labels: np.ndarray) -> float:
        call_log.append((preds, labels))
        return 0.42

    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation()],
        config=sample_config,
        custom_metrics={"my_custom": _my_metric},
    )

    # Force prediction caching by running _evaluate with cache_predictions=True
    metrics = trainer._evaluate(dummy_dataloader, cache_predictions=True)

    # Custom metric should have been computed
    assert "my_custom" in metrics
    assert metrics["my_custom"] == pytest.approx(0.42)
    # Should have received valid arrays
    assert len(call_log) == 1
    preds, labels = call_log[0]
    assert isinstance(preds, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert preds.shape == labels.shape
