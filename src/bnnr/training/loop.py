"""Training loop helpers — epoch train/evaluate and run orchestration."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bnnr.adapter import XAICapableModel
from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import BaseAugmentation
from bnnr.training import branching as _branching
from bnnr.training import callbacks as _callbacks
from bnnr.training import checkpoint as _ckpt
from bnnr.training import dataset_profile as _dprofile
from bnnr.training import metrics as _metrics
from bnnr.training import probe as _probe
from bnnr.training import xai_runner as _xai
from bnnr.training.metrics import average_metrics
from bnnr.utils import set_seed

if TYPE_CHECKING:
    from bnnr.reporting import BNNRRunResult
    from bnnr.trainer import BNNRTrainer

def train_epoch(
    trainer: BNNRTrainer,
    loader: DataLoader,
    augmentations: list[BaseAugmentation] | None = None,
) -> dict[str, float]:
    """Run one training epoch (delegates augmentation application to *trainer*)."""
    epoch_metrics: list[dict[str, float]] = []

    if trainer._is_detection:
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, targets, sample_indices = raw_batch
            else:
                images, targets = raw_batch
                sample_indices = None
            batch: Any = (images, targets)
            if augmentations:
                for aug in augmentations:
                    batch = trainer._apply_augmentation_to_batch(
                        batch, aug, sample_indices=sample_indices,
                    )
            metrics = trainer.model.train_step(batch)
            epoch_metrics.append(metrics)
        return average_metrics(epoch_metrics)

    if augmentations:
        runner = AugmentationRunner(augmentations, async_prefetch=False)
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, labels, sample_indices = raw_batch
            else:
                images, labels = raw_batch
                sample_indices = None
            images, labels = runner.apply_batch(images, labels, sample_indices=sample_indices)
            metrics = trainer.model.train_step((images, labels))
            epoch_metrics.append(metrics)
    else:
        for raw_batch in loader:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
                batch = (images, labels)
            else:
                batch = raw_batch
            metrics = trainer.model.train_step(batch)
            epoch_metrics.append(metrics)
    return average_metrics(epoch_metrics)



def evaluate(
    trainer: BNNRTrainer,
    loader: DataLoader,
    *,
    cache_predictions: bool = False,
) -> dict[str, float]:
    """Evaluate the model on *loader*."""
    all_metrics: list[dict[str, float]] = []

    can_cache = (
        cache_predictions
        and not trainer._is_detection
        and isinstance(trainer.model, XAICapableModel)
    )
    preds_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    _captured_logits: list[torch.Tensor] = []
    _hook_handle = None
    if can_cache:
        model_impl = trainer.model.get_model()  # type: ignore[attr-defined]
        model_impl.eval()

        def _capture_hook(_module: Any, _inp: Any, output: Any) -> None:
            _captured_logits.append(output.detach())

        _hook_handle = model_impl.register_forward_hook(_capture_hook)

    try:
        for raw_batch in loader:
            if trainer._is_detection:
                if len(raw_batch) == 3:
                    images, targets, _ = raw_batch
                else:
                    images, targets = raw_batch
                batch: Any = (images, targets)
                all_metrics.append(trainer.model.eval_step(batch))
                continue

            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
                batch = (images, labels)
            else:
                batch = raw_batch
                images, labels = raw_batch[0], raw_batch[1]

            _captured_logits.clear()
            all_metrics.append(trainer.model.eval_step(batch))

            if can_cache and _captured_logits:
                logits = _captured_logits[-1]
                if trainer._is_multilabel:
                    preds_rows.append(
                        (torch.sigmoid(logits) >= trainer.config.multilabel_threshold).int().cpu()
                    )
                else:
                    preds_rows.append(torch.argmax(logits, dim=1).cpu())
                label_rows.append(labels.cpu())
    finally:
        if _hook_handle is not None:
            _hook_handle.remove()

    if can_cache and preds_rows:
        trainer._last_eval_preds = torch.cat(preds_rows).numpy().astype("int64")
        trainer._last_eval_labels = torch.cat(label_rows).numpy().astype("int64")
    else:
        trainer._last_eval_preds = None
        trainer._last_eval_labels = None

    epoch_end_eval_fn = getattr(trainer.model, "epoch_end_eval", None)
    if trainer._is_detection and callable(epoch_end_eval_fn):
        epoch_level_metrics = epoch_end_eval_fn()
        avg = average_metrics(all_metrics)
        avg.pop("loss", None)
        avg.update(epoch_level_metrics)
        result = avg
    else:
        result = average_metrics(all_metrics)

    if (
        trainer._custom_metrics
        and trainer._last_eval_preds is not None
        and trainer._last_eval_labels is not None
    ):
        for name, fn in trainer._custom_metrics.items():
            try:
                result[name] = float(fn(trainer._last_eval_preds, trainer._last_eval_labels))
            except Exception as exc:  # noqa: BLE001
                trainer.logger.warning(
                    "Custom metric %r failed and was omitted from results: %s",
                    name,
                    exc,
                    exc_info=True,
                )

    return result


def run_single_iteration(
    trainer: BNNRTrainer,
    augmentation: BaseAugmentation,
    baseline_metrics: dict[str, float] | None = None,
    *,
    iteration: int = 0,
    candidate_idx: int = 0,
    total_candidates: int = 0,
    ) -> tuple[dict[str, float], dict[str, Any], int, bool]:
    """Train one candidate augmentation for m_epochs.

    Returns:
        best_metrics: metrics from the best epoch (per selection_metric).
        best_model_state: deep-copied model state_dict from the best epoch.
        best_epoch: 1-based index of the best epoch.
        pruned: whether the candidate was pruned early.
    """
    active = trainer._active_augmentations + [augmentation]
    sel_m = trainer.config.selection_metric
    sel_mode = trainer.config.selection_mode

    # Track best checkpoint across all epochs
    best_metrics: dict[str, float] = {}
    # Pre-allocate state buffer; updated in-place on each new best.
    best_model_state = trainer._clone_state_dict(trainer.model.state_dict())
    best_epoch: int = 0
    best_sel_value: float | None = None

    pruned = False
    for epoch_idx in range(1, trainer.config.m_epochs + 1):
        train_metrics = train_epoch(trainer, trainer.train_loader, augmentations=active)
        epoch_metrics = evaluate(trainer, trainer.val_loader)

        # Preserve training loss (eval pops dummy loss for detection).
        if "loss" not in epoch_metrics and "loss" in train_metrics:
            epoch_metrics["loss"] = train_metrics["loss"]

        sel_v = epoch_metrics.get(sel_m, 0)

        # Check if this epoch is the new best
        is_new_best = False
        if best_sel_value is None:
            is_new_best = True
        elif sel_mode == "max" and sel_v > best_sel_value:
            is_new_best = True
        elif sel_mode == "min" and sel_v < best_sel_value:
            is_new_best = True

        if is_new_best:
            best_metrics = epoch_metrics
            trainer._copy_state_dict_inplace(best_model_state, trainer.model.state_dict())
            best_epoch = epoch_idx
            best_sel_value = sel_v

        # Step LR scheduler if the adapter supports it
        epoch_end_fn = getattr(trainer.model, "epoch_end", None)
        if callable(epoch_end_fn):
            epoch_end_fn()

        # Print progress to terminal
        best_marker = " ★" if is_new_best else ""
        trainer.console.print(
            f"    epoch {epoch_idx}/{trainer.config.m_epochs} "
            f"— {sel_m}={sel_v:.4f}  loss={epoch_metrics.get('loss', 0):.4f}"
            f"  (best: e{best_epoch}={best_sel_value:.4f}){best_marker}",
            flush=True,
        )

        # Emit per-epoch event so the dashboard updates during candidate evaluation
        log_cand_epoch = getattr(trainer.reporter, "log_candidate_epoch", None)
        if callable(log_cand_epoch):
            log_cand_epoch(
                iteration=iteration,
                epoch=epoch_idx,
                augmentation_name=augmentation.name,
                metrics=epoch_metrics,
                is_best=is_new_best,
            )

        trainer._check_pause()

        # Pruning: use best-so-far metrics (not current epoch)
        # If even the best performance so far is below threshold, prune.
        if (
            baseline_metrics is not None
            and epoch_idx >= trainer.config.candidate_pruning_warmup_epochs
            and _branching.should_prune_candidate(best_metrics, baseline_metrics, trainer.config)
        ):
            pruned = True
            trainer.console.print(
                f"    ✗ Pruned at epoch {epoch_idx} "
                f"(best-so-far {sel_m}={best_sel_value:.4f} below threshold)",
                flush=True,
            )
            break
    return best_metrics, best_model_state, best_epoch, pruned


# Projected branch-search time above which a one-time long-run warning is printed.
_LONG_RUN_WARN_SECONDS = 3600.0


def _estimate_remaining_seconds(
    avg_candidate_seconds: float,
    candidates_per_iteration: int,
    completed_in_iteration: int,
    current_iteration: int,
    max_iterations: int,
) -> float:
    """Project wall-clock seconds left in the branch search.

    Uses the running average candidate duration and counts the unfinished
    candidates in the current iteration plus a full candidate sweep for each
    iteration still to come. Best-effort (ignores pruning/early-stop), so it is
    an upper-ish bound used only to warn about very long runs.
    """
    remaining_this_iter = max(candidates_per_iteration - completed_in_iteration, 0)
    remaining_iterations = max(max_iterations - current_iteration, 0)
    total_candidates_left = remaining_this_iter + candidates_per_iteration * remaining_iterations
    return max(avg_candidate_seconds, 0.0) * total_candidates_left


def run(trainer: BNNRTrainer) -> BNNRRunResult:
    from bnnr.reporting import BNNRRunResult

    set_seed(trainer.config.seed)
    trainer.reporter.start(trainer.config)

    # Emit dataset profile before any training (near-zero cost)
    trainer._emit_pipeline_phase("dataset_profiling", "started", "Analyzing dataset...")
    dataset_profile = _dprofile.compute_dataset_profile(trainer.train_loader, trainer.val_loader, trainer.config, is_detection=trainer._is_detection, is_multilabel=trainer._is_multilabel, reporter=trainer.reporter, log_fn=trainer._log, logger=trainer.logger)
    log_dataset_profile = getattr(trainer.reporter, "log_dataset_profile", None)
    if callable(log_dataset_profile):
        log_dataset_profile(dataset_profile)
    trainer._emit_pipeline_phase("dataset_profiling", "completed")

    baseline_metrics: dict[str, float]
    best_metrics: dict[str, float]
    if trainer.current_iteration > 0:
        trainer._log(f"Resuming from iteration {trainer.current_iteration}")
        baseline_metrics = copy.deepcopy(trainer._resume_baseline_metrics or evaluate(trainer, trainer.val_loader))
        best_metrics = baseline_metrics
        best_path = " -> ".join([aug.name for aug in trainer._active_augmentations]) or "baseline"
        selected_augmentations = [aug.name for aug in trainer._active_augmentations]
    else:
        # Baseline phase
        trainer.console.print(
            f"\n{'='*60}\n"
            f"  BASELINE TRAINING ({trainer.config.m_epochs} epochs)\n"
            f"  Starting...\n"
            f"{'='*60}",
            flush=True,
        )
        # Track the best baseline epoch by the selection metric, mirroring
        # run_single_iteration. Without this the baseline kept its last epoch
        # while candidates kept their best, biasing candidate-vs-baseline
        # deltas upward (best-vs-last).
        sel_mode = trainer.config.selection_mode
        best_baseline_state = trainer._clone_state_dict(trainer.model.state_dict())
        best_baseline_xai_stats = copy.deepcopy(trainer._prev_xai_batch_stats)
        best_baseline_epoch = 0
        best_baseline_sel: float | None = None
        for epoch in range(1, trainer.config.m_epochs + 1):
            train_metrics = train_epoch(trainer, trainer.train_loader, augmentations=[])
            val_metrics = evaluate(trainer, trainer.val_loader, cache_predictions=True)

            # Step LR scheduler if the adapter supports it
            epoch_end_fn = getattr(trainer.model, "epoch_end", None)
            if callable(epoch_end_fn):
                epoch_end_fn()

            sel_m = trainer.config.selection_metric
            # For detection, loss comes from train_metrics (eval has no loss)
            display_loss = val_metrics.get("loss", train_metrics.get("loss", 0))
            trainer.console.print(
                f"  baseline epoch {epoch}/{trainer.config.m_epochs} "
                f"— {sel_m}={val_metrics.get(sel_m, 0):.4f}  "
                f"loss={display_loss:.4f}",
                flush=True,
            )
            per_class_accuracy, confusion = _metrics.compute_eval_class_details(trainer)
            cp = _ckpt.save_checkpoint(trainer, 0, "baseline", val_metrics)

            # Generate XAI and previews BEFORE emitting events so
            # dashboard receives complete data in a single flush.
            xai_paths, xai_insights, xai_diagnoses = _xai.generate_xai(trainer,
                0, "baseline", confusion=confusion,
            )
            preview_pairs = _xai.generate_augmentation_previews(trainer, 0, "baseline", [])

            # Inject mean XAI quality score as a trackable metric
            epoch_metrics = {**train_metrics, **val_metrics}
            _xai_q = _callbacks.xai_mean_quality(xai_diagnoses)
            if _xai_q is not None:
                epoch_metrics["xai_quality"] = round(_xai_q, 4)

            # Record the best baseline epoch (state + metrics) by selection metric.
            sel_v = epoch_metrics.get(sel_m, 0)
            is_new_baseline_best = (
                best_baseline_sel is None
                or (sel_mode == "max" and sel_v > best_baseline_sel)
                or (sel_mode == "min" and sel_v < best_baseline_sel)
            )
            if is_new_baseline_best:
                # Fresh clone (not in-place copy): a YOLO model fuses conv+BN
                # during the first epoch, changing its key set, so reusing the
                # pre-fusion buffer would keep stale BN keys and break a strict
                # load_state_dict later.
                best_baseline_state = trainer._clone_state_dict(trainer.model.state_dict())
                best_baseline_xai_stats = copy.deepcopy(trainer._prev_xai_batch_stats)
                best_baseline_epoch = epoch
                best_baseline_sel = sel_v

            # Merge XAI insights into per-class data
            for cls_id, insight_text in xai_insights.items():
                if cls_id in per_class_accuracy:
                    per_class_accuracy[cls_id]["xai_insight"] = insight_text  # type: ignore[assignment]

            # Emit epoch_end with XAI data included
            log_epoch_metrics = getattr(trainer.reporter, "log_epoch_metrics", None)
            if callable(log_epoch_metrics):
                log_epoch_metrics(
                    iteration=0,
                    epoch=epoch,
                    branch="baseline",
                    metrics=epoch_metrics,
                    active_path="baseline",
                    per_class_accuracy=per_class_accuracy,
                    confusion=confusion,
                    xai_insights=xai_insights,
                    xai_diagnoses=xai_diagnoses,
                )
            trainer.reporter.log_checkpoint(
                0,
                "baseline",
                epoch,
                epoch_metrics,
                cp,
                xai_paths,
                preview_pairs=preview_pairs,
                probe_labels=_probe.probe_labels_from_tensor(trainer._report_probe_labels),
                active_path="baseline",
                per_class_accuracy=per_class_accuracy,
                confusion=confusion,
                xai_insights=xai_insights,
                xai_diagnoses=xai_diagnoses,
                emit_epoch_event=False,
            )
            # Emit sample predictions ONCE with all artifacts
            _probe.emit_probe_prediction_snapshots(trainer,
                iteration=0,
                epoch=epoch,
                branch="baseline",
                preview_pairs=preview_pairs,
                xai_paths=xai_paths,
            )
            trainer._check_pause()
        # Restore the best-epoch weights so subsequent iterations branch from
        # the best baseline (not the last epoch), then re-evaluate to refresh
        # the cached predictions used by the report.
        trainer.model.load_state_dict(best_baseline_state)
        if best_baseline_epoch and best_baseline_epoch != trainer.config.m_epochs:
            trainer.console.print(
                f"  (baseline best epoch: e{best_baseline_epoch})",
                flush=True,
            )
        baseline_metrics = evaluate(trainer, trainer.val_loader, cache_predictions=True)
        # Overwrite the on-disk baseline checkpoint (per-epoch saves left the
        # last epoch) so resume restores the best baseline, not the last one.
        _ckpt.save_checkpoint(trainer, 0, "baseline", baseline_metrics)
        best_metrics = baseline_metrics
        best_path = "baseline"
        selected_augmentations = []
        # Freeze the best-epoch baseline XAI stats for delta-vs-baseline in all
        # future checkpoints.
        trainer._baseline_xai_stats = copy.deepcopy(best_baseline_xai_stats)

        sel_m = trainer.config.selection_metric
        trainer.console.print(
            f"\n  ✓ Baseline complete: {sel_m}={baseline_metrics.get(sel_m, 0):.4f}\n",
            flush=True,
        )

        # Emit XAI-guided augmentation hints after baseline
        if xai_diagnoses and trainer._prev_xai_batch_stats:
            _xai.generate_augmentation_hints(trainer,
                xai_diagnoses, trainer._prev_xai_batch_stats, phase="baseline",
            )

    initial_baseline_metrics = copy.deepcopy(baseline_metrics)
    patience_count = 0
    best_state = trainer._clone_state_dict(trainer.model.state_dict())
    current_branch_id = "root:baseline"
    start_iteration = max(1, trainer.current_iteration or 1)
    resume_iteration = trainer.current_iteration if trainer.current_iteration > 0 else None

    # XAI cache is precomputed AFTER the baseline phase so saliency maps reflect the
    # trained baseline model rather than random initial weights. It is stored under the
    # run directory (never reused across runs) and computed once for all iterations,
    # matching the equal-compute benchmark protocol.
    trainer._emit_pipeline_phase("xai_cache", "started", "Precomputing XAI cache...")
    xai_cache = _xai.precompute_xai_cache(trainer)
    if xai_cache is None:
        trainer._emit_pipeline_phase("xai_cache", "skipped")
    else:
        trainer._emit_pipeline_phase("xai_cache", "completed")

    long_run_warned = False
    for iteration in tqdm(
        range(start_iteration, trainer.config.max_iterations + 1),
        desc="BNNR iterations",
        disable=not trainer.config.verbose,
    ):
        trainer.current_iteration = iteration
        iteration_results: dict[str, dict[str, float]]
        completed_candidates: list[str]

        if resume_iteration is not None and iteration == resume_iteration:
            iteration_results = copy.deepcopy(trainer._resume_iteration_results)
            completed_candidates = list(trainer._resume_completed_candidates)
        else:
            iteration_results = {}
            completed_candidates = []

        candidates = [a for a in trainer.augmentations if a.name not in [x.name for x in trainer._active_augmentations]]
        if completed_candidates:
            completed_set = set(completed_candidates)
            candidates = [a for a in candidates if a.name not in completed_set]
        if not candidates and not iteration_results:
            trainer._log("No remaining augmentation candidates; stopping")
            break

        # Store each candidate's best-epoch model state so we can use the winner directly
        candidate_states: dict[str, dict[str, Any]] = {}
        candidate_best_epochs: dict[str, int] = {}

        # ── Optional baseline re-evaluation for this iteration ────
        if trainer.config.reeval_baseline_per_iteration and iteration > 0:
            trainer.console.print(
                f"\n  ↻ Baseline re-evaluation (iteration {iteration}, "
                f"{trainer.config.m_epochs} epochs, no augmentations)",
                flush=True,
            )
            saved_state = trainer._clone_state_dict(trainer.model.state_dict())
            trainer.model.load_state_dict(trainer._clone_state_dict(best_state))

            # Create a no-op augmentation list for the baseline re-evaluation
            from bnnr.augmentations import BasicAugmentation
            noop_aug = BasicAugmentation(probability=1.0)
            reeval_metrics, reeval_state, reeval_epoch, _ = run_single_iteration(trainer,
                noop_aug,
                baseline_metrics=baseline_metrics,
                iteration=iteration,
                candidate_idx=0,
                total_candidates=len(candidates) + 1,
            )

            # Emit as a regular branch so the dashboard picks it up
            reeval_branch_id = f"iter_{iteration}:baseline_reeval"
            _sink = getattr(trainer.reporter, "_event_sink", None)
            if _sink is not None and hasattr(_sink, "emit"):
                _sink.emit(
                    "branch_evaluated",
                    {
                        "branch_id": reeval_branch_id,
                        "branch": "baseline_reeval",
                        "augmentation_name": "baseline_reeval",
                        "iteration": iteration,
                        "metrics": reeval_metrics,
                        "epoch": reeval_epoch,
                        "selected": False,
                    },
                )
            iteration_results["baseline_reeval"] = reeval_metrics
            candidate_states["baseline_reeval"] = reeval_state
            candidate_best_epochs["baseline_reeval"] = reeval_epoch

            # Restore original model state
            trainer.model.load_state_dict(saved_state)
            trainer.console.print(
                f"    ✓ Baseline re-eval: "
                f"{trainer.config.selection_metric}={reeval_metrics.get(trainer.config.selection_metric, 0):.4f}",
                flush=True,
            )

        if candidates:
            sel_m = trainer.config.selection_metric
            base_val = baseline_metrics.get(sel_m, 0)
            trainer.console.print(
                f"\n{'='*60}\n"
                f"  ITERATION {iteration} — Evaluating {len(candidates)} candidates\n"
                f"  Baseline {sel_m}: {base_val:.4f}\n"
                f"  Epochs per candidate: {trainer.config.m_epochs}\n"
                f"  Selection: best epoch per candidate (not last)\n"
                f"{'='*60}",
                flush=True,
            )

            candidate_bar = tqdm(
                candidates,
                desc=f"Iteration {iteration} candidates",
                leave=False,
                disable=not trainer.config.verbose,
            )
            per_candidate_durations: list[float] = []
            per_class_by_candidate: dict[str, dict[str, dict[str, float | int]]] = {}
            xai_scores_by_candidate: dict[str, float] = {}
            for idx, augmentation in enumerate(candidate_bar, start=1):
                t0 = time.perf_counter()
                trainer.console.print(
                    f"\n  ▶ [{idx}/{len(candidates)}] {augmentation.name} "
                    f"(p={augmentation.probability:.2f})",
                    flush=True,
                )
                trainer.model.load_state_dict(trainer._clone_state_dict(best_state))
                cand_best_metrics, cand_best_state, cand_best_epoch, pruned = run_single_iteration(trainer,
                    augmentation,
                    baseline_metrics=baseline_metrics,
                    iteration=iteration,
                    candidate_idx=idx,
                    total_candidates=len(candidates),
                )
                iteration_results[augmentation.name] = cand_best_metrics
                # Save this candidate's best-epoch model state (already deep-copied)
                candidate_states[augmentation.name] = cand_best_state
                candidate_best_epochs[augmentation.name] = cand_best_epoch

                # Restore best-epoch state to compute per-class details at that point.
                # Clear cached eval data so _compute_eval_class_details_detection
                # runs a fresh (single) forward pass with the best-epoch weights.
                trainer.model.load_state_dict(cand_best_state)
                if hasattr(trainer.model, "last_eval_preds"):
                    trainer.model.last_eval_preds = []  # type: ignore[attr-defined]  # duck-typed attr on DetectionAdapter
                    trainer.model.last_eval_targets = []  # type: ignore[attr-defined]  # duck-typed attr on DetectionAdapter
                # Invalidate classification prediction cache (state changed)
                trainer._last_eval_preds = None
                trainer._last_eval_labels = None
                per_class_candidate, confusion_candidate = _metrics.compute_eval_class_details(trainer)
                per_class_by_candidate[augmentation.name] = per_class_candidate

                # Lightweight XAI probe per candidate (for XAI-aware selection)
                _, cand_xai_diag, _ = _xai.generate_xai_lightweight(trainer,
                    iteration, augmentation.name, confusion=confusion_candidate,
                )
                if cand_xai_diag:
                    avg_q = float(np.mean([
                        d.get("quality_score", 0.0) for d in cand_xai_diag.values()
                    ]))
                    xai_scores_by_candidate[augmentation.name] = avg_q

                completed_candidates.append(augmentation.name)

                # Emit real-time events per candidate so dashboard updates live
                branch_id = f"iter_{iteration}:{augmentation.name}"
                trainer.reporter.log_candidate_evaluated(
                    iteration=iteration,
                    branch_id=branch_id,
                    parent_id=current_branch_id,
                    augmentation_name=augmentation.name,
                    metrics=cand_best_metrics,
                    pruned=pruned,
                    per_class=per_class_candidate,
                    confusion=confusion_candidate,
                    best_epoch=cand_best_epoch,
                    candidate_idx=idx,
                    total_candidates=len(candidates),
                )

                delta = cand_best_metrics.get(sel_m, 0) - base_val
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                status = "PRUNED" if pruned else f"Δ{delta_str}"
                trainer.console.print(
                    f"  ◀ [{idx}/{len(candidates)}] {augmentation.name}: "
                    f"{sel_m}={cand_best_metrics.get(sel_m, 0):.4f} "
                    f"(best@e{cand_best_epoch}, {status})",
                    flush=True,
                )

                elapsed = time.perf_counter() - t0
                per_candidate_durations.append(elapsed)
                avg_time = sum(per_candidate_durations) / len(per_candidate_durations)
                remaining = max(len(candidates) - idx, 0)
                eta = avg_time * remaining
                candidate_bar.set_postfix_str(f"avg={avg_time:.2f}s eta={eta:.1f}s")

                if not long_run_warned:
                    projected = _estimate_remaining_seconds(
                        avg_time,
                        len(candidates),
                        idx,
                        iteration,
                        trainer.config.max_iterations,
                    )
                    if projected >= _LONG_RUN_WARN_SECONDS:
                        long_run_warned = True
                        trainer.console.print(
                            f"\n  WARNING: branch search may take ~{projected / 3600:.1f}h more "
                            f"(~{avg_time:.0f}s/candidate over the remaining iterations). "
                            "Lower --max-iterations or pick a lighter preset to shorten it; "
                            "training is checkpointed, so Ctrl+C and resume is safe.\n",
                            flush=True,
                        )

                if trainer.config.save_checkpoints:
                    _ = _ckpt.save_checkpoint(trainer,
                        iteration=iteration,
                        augmentation_name=f"progress_{augmentation.name}",
                        metrics=cand_best_metrics,
                        baseline_metrics=baseline_metrics,
                        completed_candidates=completed_candidates,
                        current_best_metric=_branching.get_current_best_metric(iteration_results, trainer.config),
                        iteration_results=iteration_results,
                    )

                trainer._check_pause()

        selected_name = trainer._select_best_path(
            iteration_results,
            baseline_metrics,
            xai_scores=xai_scores_by_candidate if candidates else None,
        )
        top_candidate_names = _branching.top_k_candidate_names(iteration_results, trainer.config, k=3)
        candidate_preview_pairs: dict[str, list[tuple[Path, Path]]] = {}
        aug_by_name = {aug.name: aug for aug in trainer.augmentations}
        # baseline_reeval (and any non-augmentation result) is a comparison point,
        # not a selectable candidate. If it wins the iteration, no augmentation beat
        # plain continued training, so treat it as no improvement rather than looking
        # it up as an augmentation (which previously raised StopIteration).
        if selected_name is not None and selected_name not in aug_by_name:
            selected_name = None
        for cand_name in top_candidate_names:
            candidate_aug = aug_by_name.get(cand_name)
            if candidate_aug is None:
                continue
            preview_pairs = _xai.generate_augmentation_previews(trainer,
                iteration=iteration,
                augmentation_name=f"candidate_{cand_name}",
                augmentations=trainer._active_augmentations + [candidate_aug],
            )
            candidate_preview_pairs[cand_name] = preview_pairs
        trainer.reporter.log_iteration_summary(
            iteration,
            iteration_results,
            selected_name or "none",
            baseline_metrics=baseline_metrics,
            top_candidates=top_candidate_names,
            candidate_preview_pairs=candidate_preview_pairs,
            parent_branch_id=current_branch_id,
            metrics_per_class=per_class_by_candidate if candidates else {},
        )

        if selected_name is None:
            patience_count += 1
            trainer.console.print(
                f"\n  ⚠ No improvement at iteration {iteration} "
                f"({patience_count}/{trainer.config.early_stopping_patience})",
                flush=True,
            )
            if patience_count >= trainer.config.early_stopping_patience:
                trainer.console.print("  ⛔ Early stopping triggered", flush=True)
                break
            continue

        patience_count = 0
        selected_aug = next(a for a in trainer.augmentations if a.name == selected_name)
        trainer._active_augmentations.append(selected_aug)
        selected_augmentations.append(selected_name)
        trainer.best_augmentation = selected_name
        current_branch_id = f"iter_{iteration}:{selected_name}"

        # Use the winner's best-epoch model state directly
        # (already saved from the epoch with the highest selection metric)
        winner_state = candidate_states.get(selected_name)
        winner_best_epoch = candidate_best_epochs.get(selected_name, trainer.config.m_epochs)
        if winner_state is not None:
            trainer.model.load_state_dict(winner_state)
            final_metrics = iteration_results[selected_name]
        else:
            # Fallback: retrain (should not happen in normal flow)
            trainer.model.load_state_dict(trainer._clone_state_dict(best_state))
            for _ in range(trainer.config.m_epochs):
                _ = train_epoch(trainer, trainer.train_loader, augmentations=trainer._active_augmentations)
            final_metrics = evaluate(trainer, trainer.val_loader, cache_predictions=True)

        trainer._copy_state_dict_inplace(best_state, trainer.model.state_dict())
        best_metrics = final_metrics
        best_path = " -> ".join(selected_augmentations)
        baseline_metrics = final_metrics

        winner_metric = final_metrics.get(trainer.config.selection_metric, 0)
        base_val = initial_baseline_metrics.get(trainer.config.selection_metric, 0)
        trainer.console.print(
            f"\n  ★ SELECTED: '{selected_name}' "
            f"(best@epoch {winner_best_epoch}/{trainer.config.m_epochs}, "
            f"{trainer.config.selection_metric}={winner_metric:.4f}, "
            f"total gain vs initial baseline: "
            f"{'+' if winner_metric > base_val else ''}{(winner_metric - base_val):.4f})\n"
            f"  Path: {best_path}",
            flush=True,
        )

        cp = _ckpt.save_checkpoint(trainer,
            iteration,
            selected_name,
            final_metrics,
            baseline_metrics=baseline_metrics,
            completed_candidates=completed_candidates,
            current_best_metric=_branching.get_current_best_metric(iteration_results, trainer.config),
            iteration_results=iteration_results,
        )
        per_class_accuracy, confusion = _metrics.compute_eval_class_details(trainer)

        # Generate XAI and previews BEFORE emitting events
        xai_paths, xai_insights, xai_diagnoses = _xai.generate_xai(trainer,
            iteration, selected_name, confusion=confusion,
        )
        preview_pairs = _xai.generate_augmentation_previews(trainer,
            iteration,
            selected_name,
            trainer._active_augmentations,
        )

        # Inject mean XAI quality score as a trackable metric
        _xai_q = _callbacks.xai_mean_quality(xai_diagnoses)
        if _xai_q is not None:
            final_metrics["xai_quality"] = round(_xai_q, 4)

        # Merge XAI insights into per-class data
        for cls_id, insight_text in xai_insights.items():
            if cls_id in per_class_accuracy:
                per_class_accuracy[cls_id]["xai_insight"] = insight_text  # type: ignore[assignment]

        # Emit epoch_end with XAI data included
        log_epoch_metrics = getattr(trainer.reporter, "log_epoch_metrics", None)
        if callable(log_epoch_metrics):
            log_epoch_metrics(
                iteration=iteration,
                epoch=trainer.config.m_epochs,
                branch=selected_name,
                metrics=final_metrics,
                active_path=best_path,
                per_class_accuracy=per_class_accuracy,
                confusion=confusion,
                xai_insights=xai_insights,
                xai_diagnoses=xai_diagnoses,
            )
        trainer.reporter.log_checkpoint(
            iteration,
            selected_name,
            trainer.config.m_epochs,
            final_metrics,
            cp,
            xai_paths,
            preview_pairs=preview_pairs,
            probe_labels=_probe.probe_labels_from_tensor(trainer._report_probe_labels),
            active_path=best_path,
            per_class_accuracy=per_class_accuracy,
            confusion=confusion,
            xai_insights=xai_insights,
            xai_diagnoses=xai_diagnoses,
            emit_epoch_event=False,
        )
        # Emit sample predictions ONCE with all artifacts
        _probe.emit_probe_prediction_snapshots(trainer,
            iteration=iteration,
            epoch=trainer.config.m_epochs,
            branch=selected_name,
            preview_pairs=preview_pairs,
            xai_paths=xai_paths,
        )

        # Emit XAI-guided augmentation hints for this iteration
        if xai_diagnoses and trainer._prev_xai_batch_stats:
            _xai.generate_augmentation_hints(trainer,
                xai_diagnoses,
                trainer._prev_xai_batch_stats,
                phase=f"iteration {iteration}",
            )

        # Adaptive ICD/AICD threshold adjustment based on XAI diagnoses
        if trainer.config.adaptive_icd_threshold and xai_diagnoses:
            trainer._adapt_icd_thresholds(xai_diagnoses)

    analysis = _metrics.compute_eval_analysis(trainer)
    analysis["xai_insights"] = trainer._build_xai_insights(
        baseline_metrics=initial_baseline_metrics,
        best_metrics=best_metrics,
        selected_augmentations=selected_augmentations,
    )
    xai_summary = _callbacks.build_xai_summary(trainer._prev_xai_batch_stats, trainer._baseline_xai_stats)
    if xai_summary:
        analysis["xai_summary"] = xai_summary
        # Print final XAI summary to terminal
        recs = xai_summary.get("recommendations", [])
        if recs:
            trend = xai_summary.get("quality_trend", "unknown")
            cov = xai_summary.get("mean_quality_coverage", 0.0)
            foc = xai_summary.get("mean_quality_focus", 0.0)
            trainer.console.print(
                f"\n  [XAI SUMMARY] trend={trend}  "
                f"coverage={cov:.1%}  focus(gini)={foc:.2f}",
                flush=True,
            )
            for rec in recs:
                trainer.console.print(f"    → {rec}", flush=True)
    dual_xai_analysis = _xai.generate_dual_xai_analysis(trainer)
    if dual_xai_analysis:
        analysis["dual_xai"] = dual_xai_analysis
    trainer._emit_pipeline_complete()
    result = trainer.reporter.finalize(
        best_path=best_path,
        best_metrics=best_metrics,
        selected_augmentations=selected_augmentations,
        analysis=analysis,
    )
    assert isinstance(result, BNNRRunResult)
    return result

