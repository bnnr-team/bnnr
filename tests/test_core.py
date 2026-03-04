"""Tests for core training loop behavior and trainer utilities."""

from __future__ import annotations

import numpy as np
import pytest

from bnnr.augmentations import BasicAugmentation
from bnnr.core import BNNRConfig, BNNRTrainer, SimpleTorchAdapter

# ---------------------------------------------------------------------------
#  Config tests
# ---------------------------------------------------------------------------


def test_bnnr_config_defaults() -> None:
    cfg = BNNRConfig()
    assert cfg.m_epochs == 5
    assert cfg.selection_metric == "accuracy"


def test_bnnr_config_new_xai_defaults() -> None:
    """New XAI config fields should have correct defaults."""
    cfg = BNNRConfig()
    assert cfg.xai_selection_weight == 0.0
    assert cfg.xai_pruning_threshold == 0.0
    assert cfg.adaptive_icd_threshold is False
    assert cfg.report_probe_images_per_class == 3


def test_bnnr_config_xai_weight_validation() -> None:
    """xai_selection_weight must be in [0, 1]."""
    BNNRConfig(xai_selection_weight=0.5)  # should not raise
    with pytest.raises(ValueError):
        BNNRConfig(xai_selection_weight=-0.1)
    with pytest.raises(ValueError):
        BNNRConfig(xai_selection_weight=1.5)


def test_bnnr_config_xai_pruning_threshold_validation() -> None:
    """xai_pruning_threshold must be in [0, 1]."""
    BNNRConfig(xai_pruning_threshold=0.3)  # should not raise
    with pytest.raises(ValueError):
        BNNRConfig(xai_pruning_threshold=-0.01)
    with pytest.raises(ValueError):
        BNNRConfig(xai_pruning_threshold=1.01)


# ---------------------------------------------------------------------------
#  _xai_mean_quality helper
# ---------------------------------------------------------------------------


def test_xai_mean_quality_empty_diagnoses() -> None:
    result = BNNRTrainer._xai_mean_quality({})
    assert result is None


def test_xai_mean_quality_single_class() -> None:
    diag = {"0": {"quality_score": 0.75, "severity": "ok"}}
    result = BNNRTrainer._xai_mean_quality(diag)
    assert result is not None
    assert abs(result - 0.75) < 1e-6


def test_xai_mean_quality_multiple_classes() -> None:
    diag = {
        "0": {"quality_score": 0.8, "severity": "ok"},
        "1": {"quality_score": 0.6, "severity": "warning"},
        "2": {"quality_score": 0.4, "severity": "critical"},
    }
    result = BNNRTrainer._xai_mean_quality(diag)
    assert result is not None
    assert abs(result - 0.6) < 1e-6


def test_xai_mean_quality_missing_score() -> None:
    """Diagnoses without quality_score should be skipped."""
    diag = {
        "0": {"quality_score": 0.9, "severity": "ok"},
        "1": {"severity": "warning"},  # no quality_score
    }
    result = BNNRTrainer._xai_mean_quality(diag)
    assert result is not None
    assert abs(result - 0.9) < 1e-6


def test_xai_mean_quality_all_missing_scores() -> None:
    """All diagnoses without quality_score → None."""
    diag = {
        "0": {"severity": "ok"},
        "1": {"severity": "warning"},
    }
    result = BNNRTrainer._xai_mean_quality(diag)
    assert result is None


# ---------------------------------------------------------------------------
#  _build_xai_summary
# ---------------------------------------------------------------------------


def test_build_xai_summary_empty(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """No XAI stats → empty summary."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    summary = trainer._build_xai_summary()
    assert summary == {}


def test_build_xai_summary_with_stats(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """With XAI batch stats → populated summary."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    # Inject mock batch stats
    trainer._prev_xai_batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.6, "spatial_coherence": 0.7,
               "edge_ratio": 0.1, "entropy": 8.0}],
        "1": [{"coverage": 0.25, "gini": 0.4, "spatial_coherence": 0.5,
               "edge_ratio": 0.2, "entropy": 10.0}],
    }
    summary = trainer._build_xai_summary()
    assert "mean_quality_coverage" in summary
    assert "mean_quality_focus" in summary
    assert "quality_trend" in summary
    assert "per_class" in summary
    assert "recommendations" in summary
    assert "0" in summary["per_class"]
    assert "1" in summary["per_class"]
    assert summary["quality_trend"] == "insufficient_data"  # no baseline stats


def test_build_xai_summary_trend_improving(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Entropy decrease from baseline → improving trend."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    trainer._baseline_xai_stats = {
        "0": [{"entropy": 12.0, "coverage": 0.15, "gini": 0.5,
               "spatial_coherence": 0.6, "edge_ratio": 0.1}],
    }
    trainer._prev_xai_batch_stats = {
        "0": [{"entropy": 7.0, "coverage": 0.10, "gini": 0.7,
               "spatial_coherence": 0.8, "edge_ratio": 0.05}],
    }
    summary = trainer._build_xai_summary()
    assert summary["quality_trend"] == "improving"


def test_build_xai_summary_recommendations_narrow(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Narrow coverage → recommends ICD."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    trainer._prev_xai_batch_stats = {
        "0": [{"coverage": 0.02, "gini": 0.9, "spatial_coherence": 0.9,
               "edge_ratio": 0.05, "entropy": 5.0}],
    }
    summary = trainer._build_xai_summary()
    assert any("ICD" in r for r in summary["recommendations"])


def test_build_xai_summary_recommendations_broad(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Broad coverage → recommends AICD."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    trainer._prev_xai_batch_stats = {
        "0": [{"coverage": 0.55, "gini": 0.2, "spatial_coherence": 0.3,
               "edge_ratio": 0.3, "entropy": 14.0}],
    }
    summary = trainer._build_xai_summary()
    assert any("AICD" in r for r in summary["recommendations"])


# ---------------------------------------------------------------------------
#  _generate_augmentation_hints
# ---------------------------------------------------------------------------


def test_augmentation_hints_empty(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """No XAI data → no hints."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    hints = trainer._generate_augmentation_hints({}, {})
    assert hints == []


def test_augmentation_hints_narrow_coverage(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Very narrow coverage → suggests ICD."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "warning", "quality_score": 0.5}}
    batch_stats = {
        "0": [{"coverage": 0.02, "gini": 0.8, "edge_ratio": 0.05,
               "spatial_coherence": 0.7}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats, phase="baseline")
    assert any("ICD" in h for h in hints)
    assert any("narrow" in h.lower() for h in hints)


def test_augmentation_hints_broad_coverage(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Broad coverage → suggests AICD."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "ok", "quality_score": 0.7}}
    batch_stats = {
        "0": [{"coverage": 0.55, "gini": 0.5, "edge_ratio": 0.05,
               "spatial_coherence": 0.6}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("AICD" in h for h in hints)
    assert any("diffuse" in h.lower() for h in hints)


def test_augmentation_hints_low_focus(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Low Gini (uniform attention) → suggests spatial discrimination."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "warning", "quality_score": 0.4}}
    batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.2, "edge_ratio": 0.05,
               "spatial_coherence": 0.6}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("Low focus" in h for h in hints)


def test_augmentation_hints_high_edge_ratio(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """High edge ratio → suggests random cropping."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "warning", "quality_score": 0.5}}
    batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.6, "edge_ratio": 0.45,
               "spatial_coherence": 0.5}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("edge" in h.lower() for h in hints)
    assert any("crop" in h.lower() for h in hints)


def test_augmentation_hints_low_coherence(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Low coherence → suggests noise augmentation."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "warning", "quality_score": 0.4}}
    batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.6, "edge_ratio": 0.05,
               "spatial_coherence": 0.15}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("coherence" in h.lower() for h in hints)
    assert any("fragmented" in h.lower() for h in hints)


def test_augmentation_hints_critical_classes(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Critical severity → mentions the class IDs."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {
        "0": {"severity": "critical", "quality_score": 0.1},
        "1": {"severity": "ok", "quality_score": 0.8},
    }
    batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.6, "edge_ratio": 0.05,
               "spatial_coherence": 0.6}],
        "1": [{"coverage": 0.15, "gini": 0.6, "edge_ratio": 0.05,
               "spatial_coherence": 0.6}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("Critical" in h and "0" in h for h in hints)


def test_augmentation_hints_extreme_focus(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Extremely high Gini → warns about tiny region reliance."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "warning", "quality_score": 0.5}}
    batch_stats = {
        "0": [{"coverage": 0.02, "gini": 0.95, "edge_ratio": 0.05,
               "spatial_coherence": 0.8}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    assert any("concentrated" in h.lower() for h in hints)


def test_augmentation_hints_no_false_positives(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
) -> None:
    """Normal saliency values → no alarming hints generated."""
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    diagnoses: dict = {"0": {"severity": "ok", "quality_score": 0.7}}
    batch_stats = {
        "0": [{"coverage": 0.15, "gini": 0.6, "edge_ratio": 0.05,
               "spatial_coherence": 0.7}],
    }
    hints = trainer._generate_augmentation_hints(diagnoses, batch_stats)
    # No extreme values → should not trigger coverage/focus/edge/coherence hints
    assert not any("narrow" in h.lower() for h in hints)
    assert not any("diffuse" in h.lower() for h in hints)
    assert not any("Low focus" in h for h in hints)
    assert not any("edge" in h.lower() and "ratio" in h.lower() for h in hints)
    assert not any("fragmented" in h.lower() for h in hints)
    assert not any("Critical" in h for h in hints)


# ---------------------------------------------------------------------------
#  Adapter / basic trainer tests
# ---------------------------------------------------------------------------


def test_simple_torch_adapter_train_step(model_adapter: SimpleTorchAdapter, dummy_dataloader) -> None:
    batch = next(iter(dummy_dataloader))
    metrics = model_adapter.train_step(batch)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_simple_torch_adapter_state_dict(model_adapter: SimpleTorchAdapter) -> None:
    state = model_adapter.state_dict()
    model_adapter.load_state_dict(state)
    assert "model" in state and "optimizer" in state


def test_bnnr_trainer_run_one_iteration(model_adapter: SimpleTorchAdapter, dummy_dataloader, sample_config) -> None:
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    result = trainer.run()
    assert result.best_path is not None
    assert isinstance(result.best_metrics, dict)


def test_resize_saliency_batch_aligns_resolution(model_adapter: SimpleTorchAdapter, dummy_dataloader, sample_config: BNNRConfig) -> None:
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=sample_config,
    )
    maps = np.random.rand(2, 8, 8).astype(np.float32)
    resized = trainer._resize_saliency_batch(maps, target_h=32, target_w=32)
    assert resized.shape == (2, 32, 32)


def test_run_single_iteration_prunes_bad_candidate(
    model_adapter: SimpleTorchAdapter,
    dummy_dataloader,
    sample_config: BNNRConfig,
    monkeypatch,
) -> None:
    pruning_config = sample_config.model_copy(update={
        "candidate_pruning_enabled": True,
        "candidate_pruning_warmup_epochs": 1,
        "candidate_pruning_relative_threshold": 0.9,
        "m_epochs": 3,
    })
    trainer = BNNRTrainer(
        model=model_adapter,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        augmentations=[BasicAugmentation(probability=0.5)],
        config=pruning_config,
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *_args, **_kwargs: {"loss": 1.0})
    monkeypatch.setattr(trainer, "_evaluate", lambda *_args, **_kwargs: {"accuracy": 0.1, "loss": 2.0})

    _, _, _, pruned = trainer.run_single_iteration(
        BasicAugmentation(probability=0.5),
        baseline_metrics={"accuracy": 1.0, "loss": 1.0},
    )
    assert pruned is True


# ---------------------------------------------------------------------------
#  _select_best_path with XAI scores
# ---------------------------------------------------------------------------


class TestSelectBestPath:
    """Tests for _select_best_path including composite XAI selection."""

    def _make_trainer(self, model_adapter, dummy_dataloader, sample_config, **overrides):
        cfg = sample_config.model_copy(update=overrides)
        return BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=cfg,
        )

    def test_pure_metric_selection_no_xai(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Without xai_selection_weight, should pick best metric only."""
        trainer = self._make_trainer(model_adapter, dummy_dataloader, sample_config)
        results = {
            "aug_a": {"accuracy": 0.80},
            "aug_b": {"accuracy": 0.90},
        }
        baseline = {"accuracy": 0.70}
        best = trainer._select_best_path(results, baseline)
        assert best == "aug_b"

    def test_pure_metric_no_improvement(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """If no candidate beats baseline, return None."""
        trainer = self._make_trainer(model_adapter, dummy_dataloader, sample_config)
        results = {"aug_a": {"accuracy": 0.50}}
        baseline = {"accuracy": 0.70}
        best = trainer._select_best_path(results, baseline)
        assert best is None

    def test_composite_selection_xai_tips_balance(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """With xai_selection_weight, XAI quality can tip the selection."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            xai_selection_weight=0.5,
        )
        # aug_a has slightly worse metric but much better XAI quality.
        # We add a third candidate so normalisation doesn't collapse to {0, 1}.
        results = {
            "aug_a": {"accuracy": 0.83},
            "aug_b": {"accuracy": 0.85},
            "aug_c": {"accuracy": 0.80},
        }
        baseline = {"accuracy": 0.70}
        xai_scores = {"aug_a": 0.95, "aug_b": 0.30, "aug_c": 0.50}
        best = trainer._select_best_path(results, baseline, xai_scores=xai_scores)
        # aug_a composite = 0.5 * 0.6 + 0.5 * 0.95 = 0.775
        # aug_b composite = 0.5 * 1.0 + 0.5 * 0.30 = 0.650
        assert best == "aug_a"

    def test_composite_xai_scores_ignored_when_zero_weight(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """With xai_selection_weight=0, XAI scores are ignored."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            xai_selection_weight=0.0,
        )
        results = {
            "aug_a": {"accuracy": 0.82},
            "aug_b": {"accuracy": 0.85},
        }
        baseline = {"accuracy": 0.70}
        xai_scores = {"aug_a": 0.99, "aug_b": 0.01}
        best = trainer._select_best_path(results, baseline, xai_scores=xai_scores)
        # Should still pick aug_b since metric is better and weight is 0
        assert best == "aug_b"

    def test_composite_still_requires_improvement_over_baseline(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Even with good composite, if no improvement over baseline → None."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            xai_selection_weight=0.5,
        )
        results = {"aug_a": {"accuracy": 0.50}}
        baseline = {"accuracy": 0.70}
        xai_scores = {"aug_a": 0.99}
        best = trainer._select_best_path(results, baseline, xai_scores=xai_scores)
        assert best is None


# ---------------------------------------------------------------------------
#  _should_prune_candidate with XAI quality
# ---------------------------------------------------------------------------


class TestShouldPruneCandidate:
    """Tests for _should_prune_candidate with optional XAI quality check."""

    def _make_trainer(self, model_adapter, dummy_dataloader, sample_config, **overrides):
        cfg = sample_config.model_copy(update=overrides)
        return BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=cfg,
        )

    def test_no_prune_good_metric(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            candidate_pruning_enabled=True,
            candidate_pruning_relative_threshold=0.9,
        )
        # Candidate is close to baseline → no prune
        assert not trainer._should_prune_candidate(
            {"accuracy": 0.95}, {"accuracy": 1.0},
        )

    def test_prune_bad_metric(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            candidate_pruning_enabled=True,
            candidate_pruning_relative_threshold=0.9,
        )
        # Candidate is way below threshold → prune
        assert trainer._should_prune_candidate(
            {"accuracy": 0.50}, {"accuracy": 1.0},
        )

    def test_prune_by_xai_quality(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Good metric but very low XAI quality → prune."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            candidate_pruning_enabled=True,
            candidate_pruning_relative_threshold=0.9,
            xai_pruning_threshold=0.3,
        )
        # Metric is fine but XAI quality is below threshold
        assert trainer._should_prune_candidate(
            {"accuracy": 0.95}, {"accuracy": 1.0},
            xai_quality=0.15,
        )

    def test_no_prune_good_xai_quality(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Good metric and good XAI quality → no prune."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            candidate_pruning_enabled=True,
            candidate_pruning_relative_threshold=0.9,
            xai_pruning_threshold=0.3,
        )
        assert not trainer._should_prune_candidate(
            {"accuracy": 0.95}, {"accuracy": 1.0},
            xai_quality=0.7,
        )

    def test_xai_prune_disabled_by_default(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Default threshold=0 → XAI quality check is skipped."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            candidate_pruning_enabled=True,
            candidate_pruning_relative_threshold=0.9,
        )
        # Even very low XAI quality should not prune when threshold is 0
        assert not trainer._should_prune_candidate(
            {"accuracy": 0.95}, {"accuracy": 1.0},
            xai_quality=0.01,
        )


# ---------------------------------------------------------------------------
#  _adapt_icd_thresholds
# ---------------------------------------------------------------------------


class TestAdaptICDThresholds:
    """Tests for adaptive ICD threshold adjustment based on XAI stats."""

    def _make_trainer(self, model_adapter, dummy_dataloader, sample_config, **overrides):
        cfg = sample_config.model_copy(update=overrides)
        return BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=cfg,
        )

    def test_no_adjustment_when_disabled(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """When adaptive_icd_threshold=False, no changes should happen."""
        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            adaptive_icd_threshold=False,
        )
        # Populate batch stats with extreme values
        trainer._prev_xai_batch_stats = {
            "0": [{"coverage": 0.01, "gini": 0.95}],
        }
        # Nothing should happen — method exits early
        trainer._adapt_icd_thresholds({})  # type: ignore[arg-type]

    def test_adjustment_increases_for_hyperfocused(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Hyper-focused (low coverage, high gini) → increase threshold."""
        import torch.nn as nn

        from bnnr.icd import ICD

        # Create an ICD augmentation with default threshold
        dummy_model = nn.Linear(1, 1)
        icd = ICD(
            model=dummy_model,
            target_layers=[dummy_model],
            threshold_percentile=75,
        )
        old_threshold = icd.threshold_percentile

        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            adaptive_icd_threshold=True,
        )
        trainer.augmentations = [icd]
        # Set hyper-focused stats
        trainer._prev_xai_batch_stats = {
            "0": [{"coverage": 0.02, "gini": 0.9}],
            "1": [{"coverage": 0.03, "gini": 0.85}],
        }
        trainer._adapt_icd_thresholds({})
        assert icd.threshold_percentile == old_threshold + 5

    def test_adjustment_decreases_for_scattered(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Too scattered (high coverage, low gini) → decrease threshold."""
        import torch.nn as nn

        from bnnr.icd import AICD

        dummy_model = nn.Linear(1, 1)
        aicd = AICD(
            model=dummy_model,
            target_layers=[dummy_model],
            threshold_percentile=75,
        )
        old_threshold = aicd.threshold_percentile

        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            adaptive_icd_threshold=True,
        )
        trainer.augmentations = [aicd]
        # Set scattered stats
        trainer._prev_xai_batch_stats = {
            "0": [{"coverage": 0.40, "gini": 0.25}],
            "1": [{"coverage": 0.50, "gini": 0.30}],
        }
        trainer._adapt_icd_thresholds({})
        assert aicd.threshold_percentile == old_threshold - 5

    def test_threshold_clamped_to_range(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Threshold should be clamped to [50, 90]."""
        import torch.nn as nn

        from bnnr.icd import ICD

        dummy_model = nn.Linear(1, 1)
        icd = ICD(
            model=dummy_model,
            target_layers=[dummy_model],
            threshold_percentile=88,  # near max
        )

        trainer = self._make_trainer(
            model_adapter, dummy_dataloader, sample_config,
            adaptive_icd_threshold=True,
        )
        trainer.augmentations = [icd]
        trainer._prev_xai_batch_stats = {
            "0": [{"coverage": 0.01, "gini": 0.95}],
        }
        trainer._adapt_icd_thresholds({})
        assert icd.threshold_percentile <= 90


# ---------------------------------------------------------------------------
#  Change 1: AugmentationRunner in _train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpochWithAugmentationRunner:
    """Verify _train_epoch uses AugmentationRunner for classification."""

    def test_train_epoch_with_augmentation_returns_metrics(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """_train_epoch with augmentations must return valid metrics."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=sample_config,
        )
        metrics = trainer._train_epoch(
            dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
        )
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_train_epoch_no_augmentation_returns_metrics(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """_train_epoch without augmentations must still return metrics."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        metrics = trainer._train_epoch(dummy_dataloader, augmentations=[])
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_train_epoch_multiple_augmentations(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """_train_epoch handles multiple augmentations via AugmentationRunner."""
        augs = [
            BasicAugmentation(probability=0.5, random_state=42),
            BasicAugmentation(probability=0.5, random_state=43),
        ]
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=augs,
            config=sample_config,
        )
        metrics = trainer._train_epoch(dummy_dataloader, augmentations=augs)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
#  Change 2: Merged _evaluate + _compute_eval_class_details
# ---------------------------------------------------------------------------


class TestMergedEvaluate:
    """Verify _evaluate caches predictions for _compute_eval_class_details."""

    def test_evaluate_without_cache(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Default _evaluate should not populate prediction cache."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        metrics = trainer._evaluate(dummy_dataloader)
        assert "loss" in metrics
        assert trainer._last_eval_preds is None
        assert trainer._last_eval_labels is None

    def test_evaluate_with_cache(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """_evaluate(cache_predictions=True) should populate prediction cache."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        metrics = trainer._evaluate(dummy_dataloader, cache_predictions=True)
        assert "loss" in metrics
        assert trainer._last_eval_preds is not None
        assert trainer._last_eval_labels is not None
        assert len(trainer._last_eval_preds) > 0

    def test_compute_eval_class_details_uses_cache(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """_compute_eval_class_details should use cached predictions."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        # Populate cache
        trainer._evaluate(dummy_dataloader, cache_predictions=True)
        assert trainer._last_eval_preds is not None

        # Call _compute_eval_class_details — should use cache, not iterate
        per_class, confusion = trainer._compute_eval_class_details()
        # Should produce valid output (may be empty if model is random)
        assert isinstance(per_class, dict)
        assert isinstance(confusion, dict)
        if confusion:
            assert "labels" in confusion
            assert "matrix" in confusion

    def test_compute_eval_class_details_cache_vs_fallback_equivalence(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """Cached and fallback paths must produce identical results."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        # Run with cache
        trainer._evaluate(dummy_dataloader, cache_predictions=True)
        per_class_cached, confusion_cached = trainer._compute_eval_class_details()

        # Clear cache → fallback
        trainer._last_eval_preds = None
        trainer._last_eval_labels = None
        per_class_fallback, confusion_fallback = trainer._compute_eval_class_details()

        assert per_class_cached == per_class_fallback
        assert confusion_cached == confusion_fallback


# ---------------------------------------------------------------------------
#  Change 3: In-place state_dict copy
# ---------------------------------------------------------------------------


class TestInPlaceStateDictCopy:
    """Verify _clone_state_dict and _copy_state_dict_inplace."""

    def test_clone_state_dict_produces_equal_weights(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        import torch

        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        original = model_adapter.state_dict()
        cloned = trainer._clone_state_dict(original)
        for key in original["model"]:
            assert torch.equal(original["model"][key], cloned["model"][key])

    def test_inplace_copy_updates_buffer(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """In-place copy must reflect new weights in the destination buffer."""
        import torch

        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        state1 = model_adapter.state_dict()
        buffer = trainer._clone_state_dict(state1)

        # Train one step to change weights
        batch = next(iter(dummy_dataloader))
        model_adapter.train_step(batch)

        state2 = model_adapter.state_dict()
        trainer._copy_state_dict_inplace(buffer, state2)

        for key in state2["model"]:
            assert torch.equal(buffer["model"][key], state2["model"][key])

    def test_inplace_copy_does_not_share_tensors(
        self, model_adapter, dummy_dataloader, sample_config,
    ) -> None:
        """After clone + inplace copy, modifying original must not affect buffer."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        state = model_adapter.state_dict()
        buffer = trainer._clone_state_dict(state)

        # Modify original
        for v in state["model"].values():
            v.zero_()

        # Buffer should still have original values (non-zero)
        any_nonzero = False
        for v in buffer["model"].values():
            if v.abs().sum() > 0:
                any_nonzero = True
        # After initialization, there should be at least some non-zero params
        assert any_nonzero

    def test_run_single_iteration_uses_inplace_copy(
        self, model_adapter, dummy_dataloader, sample_config,
        monkeypatch,
    ) -> None:
        """run_single_iteration must still return valid results with in-place copy."""
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=sample_config,
        )
        best_metrics, best_state, best_epoch, pruned = trainer.run_single_iteration(
            BasicAugmentation(probability=0.5),
            baseline_metrics={"accuracy": 0.5, "loss": 1.0},
        )
        assert isinstance(best_metrics, dict)
        assert "accuracy" in best_metrics or "loss" in best_metrics
        assert isinstance(best_state, dict)
        assert best_epoch >= 1
        assert isinstance(pruned, bool)


# ---------------------------------------------------------------------------
#  Baseline re-evaluation config + acceptance
# ---------------------------------------------------------------------------

class TestBaselineReeval:
    def test_config_accepts_reeval_baseline_per_iteration(self) -> None:
        cfg = BNNRConfig(reeval_baseline_per_iteration=True)
        assert cfg.reeval_baseline_per_iteration is True

    def test_config_default_is_false(self) -> None:
        cfg = BNNRConfig()
        assert cfg.reeval_baseline_per_iteration is False

    def test_trainer_init_with_reeval(
        self,
        model_adapter,
        dummy_dataloader,
        sample_config,
    ) -> None:
        """BNNRTrainer should accept the reeval config without errors."""
        cfg = sample_config.model_copy(update={"reeval_baseline_per_iteration": True})
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation()],
            config=cfg,
        )
        assert trainer.config.reeval_baseline_per_iteration is True
