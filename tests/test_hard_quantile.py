"""Tests for the hard-quantile robustness proxy (FIX-1-3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.config_model import BNNRConfig
from bnnr.training.hard_quantile import HARD_QUANTILE_KEYS, hard_quantile_metrics


class TestHardQuantileMetrics:
    def test_hard_quantile_isolates_the_high_loss_tail(self) -> None:
        # Ten samples; the two hardest are the only wrong ones.
        losses = np.arange(10, dtype=np.float64)
        correct = np.ones(10, dtype=bool)
        correct[8:] = False

        out = hard_quantile_metrics(losses, correct, q=0.2)
        assert out["hard_quantile_acc"] == pytest.approx(0.0)
        assert out["robustness_gap"] == pytest.approx(0.8)

    def test_uniformly_mediocre_model_has_no_gap(self) -> None:
        """The whole point: a small gap is what a model without a shortcut looks
        like, even when its accuracy is poor."""
        rng = np.random.default_rng(0)
        losses = rng.uniform(0.0, 1.0, size=400)
        correct = rng.random(400) < 0.6  # errors independent of loss rank

        out = hard_quantile_metrics(losses, correct, q=0.2)
        assert abs(out["robustness_gap"]) < 0.12

    def test_q_is_recorded_with_the_numbers_it_produced(self) -> None:
        out = hard_quantile_metrics(np.arange(10.0), np.ones(10, dtype=bool), q=0.35)
        assert out["hard_quantile_q"] == pytest.approx(0.35)
        assert set(out) == set(HARD_QUANTILE_KEYS)

    def test_q_of_one_is_the_whole_set(self) -> None:
        losses = np.arange(10, dtype=np.float64)
        correct = np.array([True] * 6 + [False] * 4)
        out = hard_quantile_metrics(losses, correct, q=1.0)
        assert out["hard_quantile_acc"] == pytest.approx(0.6)
        assert out["robustness_gap"] == pytest.approx(0.0)

    def test_count_rounds_up_so_a_tiny_set_still_gets_one_sample(self) -> None:
        out = hard_quantile_metrics(np.array([0.1, 5.0]), np.array([True, False]), q=0.2)
        assert out["hard_quantile_acc"] == pytest.approx(0.0)

    def test_ties_are_broken_by_index_not_arbitrarily(self) -> None:
        """Every loss equal: the selection must still be deterministic."""
        losses = np.ones(10)
        correct = np.array([True] * 5 + [False] * 5)
        first = hard_quantile_metrics(losses, correct, q=0.3)
        second = hard_quantile_metrics(losses, correct, q=0.3)
        assert first == second
        assert first["hard_quantile_acc"] == pytest.approx(1.0)  # lowest indices

    def test_empty_input_reports_nothing_rather_than_zero(self) -> None:
        assert hard_quantile_metrics(np.array([]), np.array([]), q=0.2) == {}

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_q_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="q must be"):
            hard_quantile_metrics(np.arange(4.0), np.ones(4, dtype=bool), q=bad)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            hard_quantile_metrics(np.arange(4.0), np.ones(3, dtype=bool), q=0.2)

    def test_integer_correct_is_accepted(self) -> None:
        out = hard_quantile_metrics(np.arange(4.0), np.array([1, 1, 0, 0]), q=0.5)
        assert out["hard_quantile_acc"] == pytest.approx(0.0)


class TestConfigKnob:
    def test_default_is_two_tenths(self) -> None:
        assert BNNRConfig().hard_quantile_q == pytest.approx(0.2)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_out_of_range_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="hard_quantile_q"):
            BNNRConfig(hard_quantile_q=bad)

    def test_one_is_allowed(self) -> None:
        assert BNNRConfig(hard_quantile_q=1.0).hard_quantile_q == pytest.approx(1.0)


class _SeparableHead(nn.Module):
    """Maps a 1-D feature to two classes, so loss ranks with the feature."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[-4.0], [4.0]]))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.shape[0], -1))


class TestEvaluateIntegration:
    """The numbers must come out of the real evaluation path, no second pass."""

    def _trainer(self, features: torch.Tensor, labels: torch.Tensor, **config_kwargs):
        from bnnr.trainer import BNNRTrainer

        model = _SeparableHead()
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device="cpu",
        )
        loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        config = BNNRConfig(m_epochs=1, max_iterations=0, **config_kwargs)
        trainer = BNNRTrainer(
            model=adapter,
            train_loader=loader,
            val_loader=loader,
            augmentations=[],
            config=config,
        )
        return trainer, loader

    def _data(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Confident-correct samples first, then samples the head gets wrong.
        easy = torch.tensor([[2.0]] * 8)
        hard = torch.tensor([[-2.0]] * 2)
        features = torch.cat([easy, hard]).unsqueeze(-1).unsqueeze(-1)
        labels = torch.ones(10, dtype=torch.long)  # class 1 everywhere
        return features, labels

    def test_metrics_appear_in_the_evaluation_result(self) -> None:
        features, labels = self._data()
        trainer, loader = self._trainer(features, labels)
        result = trainer._evaluate(loader, cache_predictions=True)

        for key in HARD_QUANTILE_KEYS:
            assert key in result

    def test_the_hard_tail_is_the_misclassified_one(self) -> None:
        features, labels = self._data()
        trainer, loader = self._trainer(features, labels)
        result = trainer._evaluate(loader, cache_predictions=True)

        assert result["hard_quantile_acc"] == pytest.approx(0.0)
        assert result["robustness_gap"] > 0.5

    def test_configured_q_reaches_the_result(self) -> None:
        features, labels = self._data()
        trainer, loader = self._trainer(features, labels, hard_quantile_q=0.5)
        result = trainer._evaluate(loader, cache_predictions=True)
        assert result["hard_quantile_q"] == pytest.approx(0.5)

    def test_absent_without_the_prediction_cache(self) -> None:
        """No cached logits means no per-sample loss, and inventing one would
        cost the second pass this is written to avoid."""
        features, labels = self._data()
        trainer, loader = self._trainer(features, labels)
        result = trainer._evaluate(loader, cache_predictions=False)
        assert "hard_quantile_acc" not in result
