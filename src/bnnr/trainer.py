"""BNNRTrainer — iterative training, evaluation, and augmentation selection."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from bnnr.adapter import (  # noqa: F401 — re-exported for backward compat
    ModelAdapter,
    SimpleTorchAdapter,
    XAICapableModel,
)
from bnnr.augmentations import BaseAugmentation
from bnnr.config_model import BNNRConfig  # noqa: F401 — re-exported for backward compat
from bnnr.console import ConsoleReporter
from bnnr.training import branching as _branching
from bnnr.training import callbacks as _callbacks
from bnnr.training import dataset_profile as _dprofile
from bnnr.training import image_utils as _img
from bnnr.training import loop as _loop
from bnnr.training import metrics as _metrics
from bnnr.training import probe as _probe
from bnnr.training.checkpoint import (  # noqa: F401 — re-exported for backward compat
    _RuntimeState,
    _TrainerState,
    clone_state_dict,
    copy_state_dict_inplace,
)
from bnnr.utils import setup_logging
from bnnr.xai_cache import XAICache

if TYPE_CHECKING:
    from bnnr.reporting import BNNRRunResult, Reporter


class BNNRTrainer:
    """Execute iterative BNNR training, evaluation, and augmentation selection.

    Coordinates baseline/candidate epochs, checkpointing, event emission,
    XAI collection, and final report artifacts while preserving the best
    path under the configured selection metric.
    """

    def __init__(
        self,
        model: ModelAdapter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        augmentations: list[BaseAugmentation],
        config: BNNRConfig,
        reporter: Reporter | None = None,
        custom_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentations = augmentations
        self.config = config
        self.console = ConsoleReporter(config.verbose)
        self._runtime = _RuntimeState()

        if reporter is None:
            from bnnr.reporting import Reporter

            reporter = Reporter(config.report_dir)
        self.reporter = reporter

        self.current_iteration = 0
        self.best_augmentation: str | None = None
        self._active_augmentations: list[BaseAugmentation] = []
        self._resume_completed_candidates: set[str] = set()
        self._resume_iteration_results: dict[str, dict[str, float]] = {}
        self._resume_baseline_metrics: dict[str, float] | None = None
        self._report_probe_images: Tensor | None = None
        self._report_probe_labels: Tensor | None = None
        self._report_probe_targets: list[dict[str, Tensor]] | None = None
        self._report_probe_sample_ids: list[str] = []
        self._latest_detection_xai_details: list[dict[str, Any]] = []
        self._fast_probe_originals: dict[tuple[int, str, int], list[Path]] = {}
        self._prev_xai_batch_stats: dict[str, list[dict[str, float]]] = {}
        self._baseline_xai_stats: dict[str, list[dict[str, float]]] = {}
        # Cache for merged _evaluate + _compute_eval_class_details (classification only)
        self._last_eval_preds: np.ndarray | None = None
        self._last_eval_labels: np.ndarray | None = None
        self.logger = setup_logging("bnnr", config.log_file, json_format=True)
        self._custom_metrics: dict[str, Any] = custom_metrics or {}

        # ── Multi-label criterion sanity check ────────────────────────
        if self._is_multilabel and isinstance(model, SimpleTorchAdapter):
            criterion = model.criterion
            criterion_name = type(criterion).__name__
            if criterion_name in {"CrossEntropyLoss", "NLLLoss"}:
                import warnings as _w
                _w.warn(
                    f"task='multilabel' but criterion is {criterion_name}. "
                    f"Multi-label classification typically requires "
                    f"BCEWithLogitsLoss. Current criterion may produce "
                    f"incorrect gradients for multi-label targets.",
                    UserWarning,
                    stacklevel=2,
                )
            if criterion_name == "BCELoss":
                import warnings as _w
                _w.warn(
                    "task='multilabel' with BCELoss detected. BCELoss requires "
                    "sigmoid-activated inputs but the adapter passes raw logits. "
                    "Use nn.BCEWithLogitsLoss instead for numerical stability.",
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def _xai_enabled(self) -> bool:
        """Check if XAI is effectively enabled (config + runtime)."""
        return self.config.xai_enabled and not self._runtime.xai_disabled

    @property
    def _is_detection(self) -> bool:
        """Check if this run is object detection task."""
        return self.config.task == "detection"

    @property
    def _is_multilabel(self) -> bool:
        """Check if this run is multi-label classification task."""
        return self.config.task == "multilabel"

    def _log(self, message: str) -> None:
        if self.config.verbose:
            self.logger.info(message)

    def _emit_pipeline_phase(
        self,
        phase: str,
        status: str,
        message: str = "",
    ) -> None:
        """Emit a ``pipeline_phase`` event so the dashboard can show pre-training activity."""
        emit_fn = getattr(self.reporter, "_event_sink", None)
        if emit_fn is not None and hasattr(emit_fn, "emit"):
            emit_fn.emit(
                "pipeline_phase",
                {"phase": phase, "status": status, "message": message},
                force=True,
            )

    def _emit_pipeline_complete(self) -> None:
        """Emit a ``pipeline_complete`` event so the dashboard knows training finished."""
        emit_fn = getattr(self.reporter, "_event_sink", None)
        if emit_fn is not None and hasattr(emit_fn, "emit"):
            emit_fn.emit("pipeline_complete", {}, force=True)

    def _check_pause(self) -> None:
        """Poll for a pause signal file and block until it is removed.

        The dashboard creates ``<run_dir>/.bnnr_pause`` when the user clicks
        Pause.  This method is called at epoch boundaries so training
        can be suspended without altering the core training logic.
        """
        from bnnr.dashboard.constants import PAUSE_SIGNAL_FILENAME

        pause_file = self.reporter.run_dir / PAUSE_SIGNAL_FILENAME
        if not pause_file.exists():
            return
        self.console.print("\n  ⏸  Training paused — waiting for resume signal ...")
        self._log("Training paused via dashboard signal")
        while pause_file.exists():
            time.sleep(0.5)
        self.console.print("  ▶  Training resumed\n")
        self._log("Training resumed")

    def _average_metrics(self, all_metrics: list[dict[str, float]]) -> dict[str, float]:
        return _metrics.average_metrics(all_metrics)

    # Delegate to module-level helpers for backward compat
    _clone_state_dict = staticmethod(clone_state_dict)
    _copy_state_dict_inplace = staticmethod(copy_state_dict_inplace)

    def _tensor_to_uint8(self, images: Tensor) -> np.ndarray:
        return _img.tensor_to_uint8(images, warn_context=self)

    @staticmethod
    def _uint8_to_tensor(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
        return _img.uint8_to_tensor(np_images, ref_batch=ref_batch)

    @staticmethod
    def _det_uint8_batch_to_float01(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
        return _img.det_uint8_batch_to_float01(np_images, ref_batch=ref_batch)

    def _apply_augmentation_to_batch(
        self,
        batch: Any,
        augmentation: BaseAugmentation,
        sample_indices: Tensor | None = None,
    ) -> Any:
        from bnnr.training import augmentation_batch as _abatch
        return _abatch.apply_augmentation_to_batch(self, batch, augmentation, sample_indices)

    def _train_epoch(self, loader: DataLoader, augmentations: list[BaseAugmentation] | None = None) -> dict[str, float]:
        return _loop.train_epoch(self, loader, augmentations)

    def _evaluate(self, loader: DataLoader, *, cache_predictions: bool = False) -> dict[str, float]:
        return _loop.evaluate(self, loader, cache_predictions=cache_predictions)

    def _save_checkpoint(
        self,
        iteration: int,
        augmentation_name: str,
        metrics: dict[str, float],
        baseline_metrics: dict[str, float] | None = None,
        completed_candidates: list[str] | None = None,
        current_best_metric: float | None = None,
        iteration_results: dict[str, dict[str, float]] | None = None,
    ) -> Path:
        from bnnr.training.checkpoint import save_checkpoint
        return save_checkpoint(self, iteration, augmentation_name, metrics, baseline_metrics, completed_candidates, current_best_metric, iteration_results)

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        from bnnr.training.checkpoint import load_checkpoint
        return load_checkpoint(self, checkpoint_path)

    def _select_best_path(
        self,
        results: dict[str, dict[str, float]],
        baseline_metrics: dict[str, float],
        xai_scores: dict[str, float] | None = None,
    ) -> str | None:
        return _branching.select_best_path(results, baseline_metrics, self.config, xai_scores)

    def _should_prune_candidate(
        self,
        candidate_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        xai_quality: float | None = None,
    ) -> bool:
        return _branching.should_prune_candidate(candidate_metrics, baseline_metrics, self.config, xai_quality)

    def _get_current_best_metric(self, results: dict[str, dict[str, float]]) -> float | None:
        return _branching.get_current_best_metric(results, self.config)

    def _top_k_candidate_names(self, results: dict[str, dict[str, float]], k: int = 3) -> list[str]:
        return _branching.top_k_candidate_names(results, self.config, k)

    def _generate_xai(
        self,
        iteration: int,
        augmentation_name: str,
        confusion: dict[str, Any] | None = None,
    ) -> tuple[list[Path], dict[str, str], dict[str, dict[str, Any]]]:
        from bnnr.training import xai_runner as _xai
        return _xai.generate_xai(self, iteration, augmentation_name, confusion)

    @staticmethod
    def _xai_mean_quality(xai_diagnoses: dict[str, dict[str, Any]]) -> float | None:
        return _callbacks.xai_mean_quality(xai_diagnoses)

    def _generate_xai_lightweight(
        self,
        iteration: int,
        augmentation_name: str,
        confusion: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
        from bnnr.training import xai_runner as _xai
        return _xai.generate_xai_lightweight(self, iteration, augmentation_name, confusion)

    def _tensor_batch_to_preview_uint8(self, images: Tensor) -> np.ndarray:
        return _img.tensor_batch_to_preview_uint8(
            images,
            denorm_mean=self.config.denormalization_mean,
            denorm_std=self.config.denormalization_std,
        )

    def _generate_augmentation_previews(
        self,
        iteration: int,
        augmentation_name: str,
        augmentations: list[BaseAugmentation] | None,
    ) -> list[tuple[Path, Path]]:
        from bnnr.training import xai_runner as _xai
        return _xai.generate_augmentation_previews(self, iteration, augmentation_name, augmentations)

    def _initialize_report_probe_samples(self) -> None:
        _probe.initialize_report_probe_samples(self)

    def _probe_labels(self) -> list[int]:
        return _probe.probe_labels_from_tensor(self._report_probe_labels)

    def _probe_sample_ids(self) -> list[str]:
        return _probe.probe_sample_ids_from_list(self._report_probe_sample_ids)

    def _collect_val_logits(
        self,
        *,
        post_process: str = "argmax",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        from bnnr.training import metrics as _metrics
        return _metrics.collect_val_logits(self, post_process=post_process)

    def _compute_eval_class_details(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        from bnnr.training import metrics as _metrics
        return _metrics.compute_eval_class_details(self)

    def _compute_eval_class_details_multilabel(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        result = _metrics.collect_val_logits(self, post_process="sigmoid")
        if result is None:
            return {}, {}
        preds, labels = result
        return _metrics.compute_multilabel_eval_details(preds, labels)

    def _compute_eval_class_details_detection(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        from bnnr.training import metrics as _metrics
        return _metrics.compute_eval_class_details_detection(self)

    def _emit_probe_prediction_snapshots(
        self,
        *,
        iteration: int,
        epoch: int,
        branch: str,
        preview_pairs: list[tuple[Path, Path]],
        xai_paths: list[Path],
    ) -> None:
        from bnnr.training import probe as _probe
        return _probe.emit_probe_prediction_snapshots(self, iteration=iteration, epoch=epoch, branch=branch, preview_pairs=preview_pairs, xai_paths=xai_paths)

    def _ensure_fast_probe_originals(
        self,
        *,
        iteration: int,
        branch: str,
        epoch: int,
        run_dir: Path | None,
    ) -> list[Path]:
        from bnnr.training import probe as _probe
        return _probe.ensure_fast_probe_originals(self, iteration=iteration, branch=branch, epoch=epoch, run_dir=run_dir)

    def _to_artifact_reference(self, path: Path, run_dir: Path | None) -> str:
        from bnnr.training.probe import to_artifact_reference
        return to_artifact_reference(path, run_dir)

    def _adapt_icd_thresholds(self, xai_diagnoses: dict[str, dict[str, Any]]) -> None:
        _callbacks.adapt_icd_thresholds(
            self.augmentations,
            self._prev_xai_batch_stats,
            adaptive_enabled=self.config.adaptive_icd_threshold,
            log_fn=self._log,
        )

    def _compute_eval_analysis(self) -> dict[str, Any]:
        from bnnr.training import metrics as _metrics
        return _metrics.compute_eval_analysis(self)

    def _build_xai_insights(
        self,
        baseline_metrics: dict[str, float],
        best_metrics: dict[str, float],
        selected_augmentations: list[str],
    ) -> list[str]:
        return _callbacks.build_xai_insights(baseline_metrics, best_metrics, selected_augmentations, self.config)

    @staticmethod
    def _saliency_recommendations(
        batch_stats: dict[str, list[dict[str, float]]],
        xai_diagnoses: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        return _callbacks.saliency_recommendations(batch_stats, xai_diagnoses)

    def _build_xai_summary(self) -> dict[str, Any]:
        return _callbacks.build_xai_summary(self._prev_xai_batch_stats, self._baseline_xai_stats)

    def _generate_augmentation_hints(
        self,
        xai_diagnoses: dict[str, dict[str, Any]],
        batch_stats: dict[str, list[dict[str, float]]],
        *,
        phase: str = "baseline",
    ) -> list[str]:
        from bnnr.training import xai_runner as _xai
        return _xai.generate_augmentation_hints(self, xai_diagnoses, batch_stats, phase=phase)

    def _resize_saliency_batch(self, maps: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        return _img.resize_saliency_batch(maps, target_h, target_w)

    def _generate_dual_xai_analysis(self) -> dict[str, Any]:
        from bnnr.training import xai_runner as _xai
        return _xai.generate_dual_xai_analysis(self)

    def _precompute_xai_cache(self) -> XAICache | None:
        from bnnr.training import xai_runner as _xai
        return _xai.precompute_xai_cache(self)

    def run_single_iteration(
        self,
        augmentation: BaseAugmentation,
        baseline_metrics: dict[str, float] | None = None,
        *,
        iteration: int = 0,
        candidate_idx: int = 0,
        total_candidates: int = 0,
    ) -> tuple[dict[str, float], dict[str, Any], int, bool]:
        from bnnr.training import loop as _loop
        return _loop.run_single_iteration(self, augmentation, baseline_metrics, iteration=iteration, candidate_idx=candidate_idx, total_candidates=total_candidates)

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        state = self._load_checkpoint(checkpoint_path)
        self.current_iteration = int(state.get("iteration", 0))
        names = state.get("active_augmentations", [])
        self._active_augmentations = [a for a in self.augmentations if a.name in names]
        self._resume_completed_candidates = set(state.get("completed_candidates", []))
        self._resume_iteration_results = cast(dict[str, dict[str, float]], state.get("iteration_results", {}))
        self._resume_baseline_metrics = cast(Optional[dict[str, float]], state.get("baseline_metrics"))

    @staticmethod
    def _count_labels(
        loader: DataLoader,
        *,
        is_detection: bool,
        is_multilabel: bool,
        capture_shape: bool = False,
    ) -> tuple[Counter[int], list[int], int]:
        return _dprofile.count_labels(
            loader,
            is_detection=is_detection,
            is_multilabel=is_multilabel,
            capture_shape=capture_shape,
        )

    def _compute_dataset_profile(self) -> dict[str, Any]:
        return _dprofile.compute_dataset_profile(
            self.train_loader,
            self.val_loader,
            self.config,
            is_detection=self._is_detection,
            is_multilabel=self._is_multilabel,
            reporter=self.reporter,
            log_fn=self._log,
            logger=self.logger,
        )

    def run(self) -> BNNRRunResult:
        from bnnr.training import loop as _loop
        return _loop.run(self)

    def to_json(self) -> str:
        state = {
            "current_iteration": self.current_iteration,
            "best_augmentation": self.best_augmentation,
            "active_augmentations": [aug.name for aug in self._active_augmentations],
        }
        return json.dumps(state)
