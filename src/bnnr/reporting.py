"""Training report dataclasses and report persistence helpers."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bnnr.core import BNNRConfig
from bnnr.events import JsonlEventSink
from bnnr.utils import ensure_dir, get_timestamp, portable_path

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    iteration: int
    augmentation: str
    epoch: int
    metrics: dict[str, float]
    checkpoint_path: Path
    xai_paths: list[Path]
    preview_pairs: list[tuple[Path, Path]]
    probe_labels: list[int]
    active_path: str
    timestamp: datetime


@dataclass
class BNNRRunResult:
    config: BNNRConfig
    checkpoints: list[CheckpointInfo]
    best_path: str
    best_metrics: dict[str, float]
    selected_augmentations: list[str]
    total_time: float
    report_json_path: Path
    report_html_path: Path | None
    analysis: dict[str, Any] = field(default_factory=dict)


class Reporter:
    """Collect run artifacts and persist structured BNNR reports.

    The reporter owns the run directory, stores checkpoint metadata,
    copies sample/XAI artifacts, emits JSONL dashboard events, and writes
    final JSON/HTML summaries for downstream CLI and dashboard usage.
    """
    def __init__(
        self,
        report_dir: Path,
        run_name: str | None = None,
        save_json: bool = True,
        save_html: bool = True,
    ) -> None:
        self.report_dir = report_dir
        self.run_name = run_name or f"run_{get_timestamp()}"
        self.run_dir = ensure_dir(self.report_dir / self.run_name)
        self.save_json = save_json
        self.save_html = save_html
        self._checkpoints: list[CheckpointInfo] = []
        self._config: BNNRConfig | None = None
        self._start_time: datetime | None = None
        self._iteration_summaries: dict[int, dict[str, Any]] = {}
        self._event_sink: JsonlEventSink | None = None
        self._probe_set_emitted = False

    def _copy_artifact_into_run_dir(self, source_path: Path, target_subdir: Path) -> Path:
        """Store an artifact in the run directory.

        Always performs a full copy (not a hard link) because source
        files in the staging area (e.g. ``report_dir/xai/``) are
        overwritten in place on every epoch.  A hard link would cause
        all per-epoch copies to share the same inode, so earlier epochs
        would silently show the latest epoch's data.
        """
        target_dir = ensure_dir(self.run_dir / target_subdir)
        target_path = target_dir / source_path.name
        # Always copy — even if target exists — because the source
        # may have been updated (e.g. new epoch wrote a new XAI map
        # with the same filename).
        shutil.copy2(source_path, target_path)
        return target_path

    def start(self, config: BNNRConfig) -> None:
        self._config = config
        self._start_time = datetime.now()
        ensure_dir(self.run_dir)
        if config.event_log_enabled:
            self._event_sink = JsonlEventSink(
                self.run_dir / "events.jsonl",
                run_id=self.run_name,
                sample_every_epochs=config.event_sample_every_epochs,
                xai_every_epochs=config.event_xai_every_epochs,
                min_interval_seconds=config.event_min_interval_seconds,
            )
            task = config.task if hasattr(config, "task") else "classification"
            if task == "detection":
                metric_units = {
                    "map_50": "%",
                    "map_50_95": "%",
                    "loss": "unitless",
                }
            else:
                metric_units = {
                    "accuracy": "%",
                    "f1_macro": "%",
                    "loss": "unitless",
                }
            self._event_sink.emit(
                "run_started",
                {
                    "run_name": self.run_name,
                    "config": config.model_dump(mode="json"),
                    "run_dir": str(self.run_dir),
                    "class_names": [],
                    "task": task,
                    "sample_every": config.event_sample_every_epochs,
                    "xai_every": config.event_xai_every_epochs,
                    "metric_units": metric_units,
                },
            )

    def log_dataset_profile(self, profile: dict[str, Any]) -> None:
        """Emit a dataset_profile event with class distribution and metadata.

        This is called once at training start and has near-zero cost.
        """
        if self._event_sink is None:
            return
        self._event_sink.emit("dataset_profile", profile)

    def log_checkpoint(
        self,
        iteration: int,
        augmentation: str,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_path: Path,
        xai_paths: list[Path] | None = None,
        preview_pairs: list[tuple[Path, Path]] | None = None,
        probe_labels: list[int] | None = None,
        active_path: str | None = None,
        per_class_accuracy: dict[str, dict[str, float | int]] | None = None,
        confusion: dict[str, Any] | None = None,
        xai_insights: dict[str, str] | None = None,
        xai_diagnoses: dict[str, dict[str, Any]] | None = None,
        emit_epoch_event: bool = True,
    ) -> None:
        # Include epoch in artifact paths so that each epoch's XAI/samples
        # are preserved separately (especially important during baseline where
        # iteration+augmentation stay the same across epochs).
        epoch_tag = f"epoch_{epoch}"
        normalized_xai_paths: list[Path] = []
        for path in xai_paths or []:
            if not path.exists():
                continue
            rel_subdir = Path("artifacts") / "xai" / f"iter_{iteration}_{augmentation}" / epoch_tag
            normalized_xai_paths.append(self._copy_artifact_into_run_dir(path, rel_subdir))

        normalized_preview_pairs: list[tuple[Path, Path]] = []
        for idx, pair in enumerate(preview_pairs or []):
            src_original, src_augmented = pair
            if not src_original.exists() or not src_augmented.exists():
                continue
            rel_subdir = Path("artifacts") / "samples" / f"iter_{iteration}_{augmentation}" / epoch_tag / f"sample_{idx}"
            original_dst = self._copy_artifact_into_run_dir(src_original, rel_subdir)
            augmented_dst = self._copy_artifact_into_run_dir(src_augmented, rel_subdir)
            normalized_preview_pairs.append((original_dst, augmented_dst))

        self._checkpoints.append(
            CheckpointInfo(
                iteration=iteration,
                augmentation=augmentation,
                epoch=epoch,
                metrics=metrics,
                checkpoint_path=checkpoint_path,
                xai_paths=normalized_xai_paths,
                preview_pairs=normalized_preview_pairs,
                probe_labels=probe_labels or [],
                active_path=active_path or augmentation,
                timestamp=datetime.now(),
            )
        )
        if self._event_sink is not None and emit_epoch_event:
            self._event_sink.emit(
                "epoch_end",
                {
                    "iteration": iteration,
                    "epoch": epoch,
                    "branch": augmentation,
                    "active_path": active_path or augmentation,
                    "metrics": metrics,
                    "per_class_accuracy": per_class_accuracy or {},
                    "confusion": confusion or {},
                    "xai_insights": xai_insights or {},
                    "xai_diagnoses": xai_diagnoses or {},
                },
            )
        if self._event_sink is not None:
            if normalized_preview_pairs:
                self._event_sink.emit(
                    "sample_snapshot",
                    {
                        "iteration": iteration,
                        "epoch": epoch,
                        "branch": augmentation,
                        "branch_id": f"iter_{iteration}:{augmentation}",
                        "probe_labels": probe_labels or [],
                        "sample_pairs": [
                            [
                                portable_path(pair[0].relative_to(self.run_dir)),
                                portable_path(pair[1].relative_to(self.run_dir)),
                            ]
                            for pair in normalized_preview_pairs
                        ],
                    },
                    force=True,  # Checkpoint snapshots must not be throttled
                )
            if normalized_xai_paths:
                self._event_sink.emit(
                    "xai_snapshot",
                    {
                        "iteration": iteration,
                        "epoch": epoch,
                        "branch": augmentation,
                        "branch_id": f"iter_{iteration}:{augmentation}",
                        "method": self._config.xai_method if self._config is not None else "unknown",
                        "artifact_paths": [portable_path(path.relative_to(self.run_dir)) for path in normalized_xai_paths],
                    },
                    force=True,  # Checkpoint snapshots must not be throttled
                )

    def log_epoch_metrics(
        self,
        *,
        iteration: int,
        epoch: int,
        branch: str,
        metrics: dict[str, float],
        active_path: str | None = None,
        is_best_epoch: bool = False,
        per_class_accuracy: dict[str, dict[str, float | int]] | None = None,
        confusion: dict[str, Any] | None = None,
        xai_insights: dict[str, str] | None = None,
        xai_diagnoses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Emit an epoch event for live dashboard updates.

        Now accepts ``xai_insights`` and ``xai_diagnoses`` so that the
        epoch_end event can carry XAI data when available.
        """
        if self._event_sink is None:
            return
        self._event_sink.emit(
            "epoch_end",
            {
                "iteration": iteration,
                "epoch": epoch,
                "branch": branch,
                "active_path": active_path or branch,
                "metrics": metrics,
                "is_best_epoch": bool(is_best_epoch),
                "per_class_accuracy": per_class_accuracy or {},
                "confusion": confusion or {},
                "xai_insights": xai_insights or {},
                "xai_diagnoses": xai_diagnoses or {},
            },
        )

    def log_probe_set(self, probes: list[dict[str, Any]]) -> None:
        if self._event_sink is None or self._probe_set_emitted:
            return
        self._event_sink.emit("probe_set_initialized", {"probes": probes})
        self._probe_set_emitted = True

    def log_sample_prediction(
        self,
        *,
        sample_id: str,
        iteration: int,
        epoch: int,
        branch: str,
        true_class: int,
        predicted_class: int,
        confidence: float,
        loss_local: float | None,
        original_artifact: str | None = None,
        augmented_artifact: str | None = None,
        xai_artifact: str | None = None,
        xai_gt_artifact: str | None = None,
        xai_saliency_artifact: str | None = None,
        xai_pred_artifact: str | None = None,
        detection_details: dict[str, Any] | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        artifacts: dict[str, str | None] = {
            "original": original_artifact,
            "augmented": augmented_artifact,
            "xai": xai_saliency_artifact or xai_artifact,
        }
        if xai_gt_artifact is not None:
            artifacts["xai_gt"] = xai_gt_artifact
        if xai_saliency_artifact is not None:
            artifacts["xai_saliency"] = xai_saliency_artifact
        if xai_pred_artifact is not None:
            artifacts["xai_pred"] = xai_pred_artifact
        self._event_sink.emit(
            "sample_prediction_snapshot",
            {
                "sample_id": sample_id,
                "branch_id": f"iter_{iteration}:{branch}",
                "iteration": iteration,
                "epoch": epoch,
                "branch": branch,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "loss_local": loss_local,
                "artifacts": artifacts,
                "detection_details": detection_details or {},
            },
        )

    def log_candidate_epoch(
        self,
        *,
        iteration: int,
        epoch: int,
        augmentation_name: str,
        metrics: dict[str, float],
        is_best: bool = False,
        per_class: dict[str, dict[str, float | int]] | None = None,
        confusion: dict[str, Any] | None = None,
    ) -> None:
        """Emit an epoch event during candidate evaluation.

        This allows the dashboard to update per-epoch instead of only when
        a candidate finishes all its epochs.
        """
        if self._event_sink is None:
            return
        self._event_sink.emit(
            "epoch_end",
            {
                "iteration": iteration,
                "epoch": epoch,
                "branch": augmentation_name,
                "active_path": f"evaluating:{augmentation_name}",
                "metrics": metrics,
                "is_best_epoch": is_best,
                "per_class_accuracy": per_class or {},
                "confusion": confusion or {},
            },
        )

    def log_candidate_evaluated(
        self,
        *,
        iteration: int,
        branch_id: str,
        parent_id: str,
        augmentation_name: str,
        metrics: dict[str, float],
        pruned: bool = False,
        per_class: dict[str, dict[str, float | int]] | None = None,
        confusion: dict[str, Any] | None = None,
        best_epoch: int = 0,
        candidate_idx: int = 0,
        total_candidates: int = 0,
    ) -> None:
        """Emit events immediately when a single candidate finishes evaluation.

        This allows the dashboard to update in real time instead of waiting
        for all candidates in an iteration to complete.

        ``best_epoch`` indicates which epoch (1-based) produced the best
        selection metric.  The reported ``metrics`` correspond to that epoch.
        """
        if self._event_sink is None:
            return
        m_epochs = self._config.m_epochs if self._config else 0
        self._event_sink.emit(
            "branch_created",
            {
                "iteration": iteration,
                "branch_id": branch_id,
                "parent_id": parent_id,
                "augmentation": augmentation_name,
                "label": augmentation_name,
                "depth": iteration,
            },
        )
        self._event_sink.emit(
            "branch_evaluated",
            {
                "iteration": iteration,
                "branch": augmentation_name,
                "branch_id": branch_id,
                "parent_id": parent_id,
                "metrics": metrics,
                "pruned": pruned,
                "best_epoch": best_epoch,
                "total_epochs": m_epochs,
                "metrics_per_class": per_class or {},
                "candidate_idx": candidate_idx,
                "total_candidates": total_candidates,
                "metric_units": (
                    {"map_50": "%", "map_50_95": "%", "loss": "unitless"}
                    if self._config is not None and getattr(self._config, "task", "classification") == "detection"
                    else {"accuracy": "%", "f1_macro": "%", "loss": "unitless"}
                ),
            },
        )
        # Also emit epoch_end so the metrics timeline updates live
        self._event_sink.emit(
            "epoch_end",
            {
                "iteration": iteration,
                "epoch": best_epoch or m_epochs,
                "branch": augmentation_name,
                "active_path": f"candidate:{augmentation_name}",
                "metrics": metrics,
                "is_best_epoch": True,
                "per_class_accuracy": per_class or {},
                "confusion": confusion or {},
            },
        )

    def log_iteration_summary(
        self,
        iteration: int,
        results: dict[str, dict[str, float]],
        selected_aug: str,
        baseline_metrics: dict[str, float] | None = None,
        top_candidates: list[str] | None = None,
        candidate_preview_pairs: dict[str, list[tuple[Path, Path]]] | None = None,
        parent_branch_id: str = "root:baseline",
        metrics_per_class: dict[str, dict[str, dict[str, float | int]]] | None = None,
    ) -> None:
        normalized_candidate_previews: dict[str, list[list[str]]] = {}
        for candidate_name, preview_pairs in (candidate_preview_pairs or {}).items():
            copied_pairs: list[list[str]] = []
            for pair_idx, pair in enumerate(preview_pairs):
                src_original, src_augmented = pair
                if not src_original.exists() or not src_augmented.exists():
                    continue
                rel_subdir = Path("artifacts") / "candidate_previews" / f"iter_{iteration}" / candidate_name / f"sample_{pair_idx}"
                original_dst = self._copy_artifact_into_run_dir(src_original, rel_subdir)
                augmented_dst = self._copy_artifact_into_run_dir(src_augmented, rel_subdir)
                copied_pairs.append([portable_path(original_dst), portable_path(augmented_dst)])
            normalized_candidate_previews[candidate_name] = copied_pairs

        self._iteration_summaries[iteration] = {
            "results": results,
            "selected": selected_aug,
            "baseline_metrics": baseline_metrics or {},
            "top_candidates": top_candidates or [],
            "candidate_previews": normalized_candidate_previews,
        }
        if self._event_sink is not None:
            # NOTE: branch_created + branch_evaluated are already emitted
            # per-candidate by log_candidate_evaluated() — only emit
            # branch_selected here to avoid duplicates.
            selected_id = f"iter_{iteration}:{selected_aug}" if selected_aug != "none" else "none"
            self._event_sink.emit(
                "branch_selected",
                {
                    "iteration": iteration,
                    "selected_branch": selected_aug,
                    "selected_branch_id": selected_id,
                    "active_path": [cp.augmentation for cp in self._checkpoints if cp.iteration >= 0],
                    "baseline_metrics": baseline_metrics or {},
                    "results": results,
                    "decision_reason": _decision_reason(
                        selected_aug,
                        results,
                        baseline_metrics or {},
                        selection_metric=self._config.selection_metric if self._config else "accuracy",
                    ),
                },
            )

    def log_message(self, message: str, level: str = "info") -> None:
        log_path = self.run_dir / "run.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}][{level.upper()}] {message}\n")

    def _generate_json_report(self, result: BNNRRunResult) -> Path:
        json_path = self.run_dir / "report.json"
        payload = {
            "config": result.config.model_dump(mode="json"),
            "best_path": result.best_path,
            "best_metrics": result.best_metrics,
            "selected_augmentations": result.selected_augmentations,
            "total_time": result.total_time,
            "checkpoints": [
                {
                    **asdict(cp),
                    "checkpoint_path": portable_path(cp.checkpoint_path),
                    "xai_paths": [portable_path(p) for p in cp.xai_paths],
                    "preview_pairs": [[portable_path(p[0]), portable_path(p[1])] for p in cp.preview_pairs],
                    "probe_labels": cp.probe_labels,
                    "active_path": cp.active_path,
                    "timestamp": cp.timestamp.isoformat(),
                }
                for cp in result.checkpoints
            ],
            "iteration_summaries": self._iteration_summaries,
            "analysis": result.analysis,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path

    def finalize(
        self,
        best_path: str,
        best_metrics: dict[str, float],
        selected_augmentations: list[str],
        analysis: dict[str, Any] | None = None,
    ) -> BNNRRunResult:
        if self._config is None or self._start_time is None:
            raise RuntimeError("Reporter.start() must be called before finalize()")

        total_time = (datetime.now() - self._start_time).total_seconds()
        placeholder_json = self.run_dir / "report.json"
        result = BNNRRunResult(
            config=self._config,
            checkpoints=self._checkpoints,
            best_path=best_path,
            best_metrics=best_metrics,
            selected_augmentations=selected_augmentations,
            total_time=total_time,
            report_json_path=placeholder_json,
            report_html_path=None,
            analysis=analysis or {},
        )

        if self.save_json:
            result.report_json_path = self._generate_json_report(result)
        if self._event_sink is not None:
            self._event_sink.close()
        return result


def load_report(report_path: Path) -> BNNRRunResult:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    cfg = BNNRConfig(**payload["config"])
    checkpoints: list[CheckpointInfo] = []
    for cp in payload["checkpoints"]:
        checkpoints.append(
            CheckpointInfo(
                iteration=cp["iteration"],
                augmentation=cp["augmentation"],
                epoch=cp["epoch"],
                metrics=cp["metrics"],
                checkpoint_path=Path(cp["checkpoint_path"]),
                xai_paths=[Path(p) for p in cp.get("xai_paths", [])],
                preview_pairs=[(Path(p[0]), Path(p[1])) for p in cp.get("preview_pairs", [])],
                probe_labels=[int(v) for v in cp.get("probe_labels", [])],
                active_path=str(cp.get("active_path", cp["augmentation"])),
                timestamp=datetime.fromisoformat(cp["timestamp"]),
            )
        )

    return BNNRRunResult(
        config=cfg,
        checkpoints=checkpoints,
        best_path=payload["best_path"],
        best_metrics=payload["best_metrics"],
        selected_augmentations=payload["selected_augmentations"],
        total_time=payload["total_time"],
        report_json_path=report_path,
        report_html_path=report_path.with_suffix(".html") if report_path.with_suffix(".html").exists() else None,
        analysis=payload.get("analysis", {}),
    )


def compare_runs(reports: list[BNNRRunResult], metrics: list[str] | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for idx, report in enumerate(reports):
        key = f"run_{idx}"
        if metrics is None:
            output[key] = report.best_metrics
        else:
            output[key] = {metric: report.best_metrics.get(metric) for metric in metrics}
    return output


def _decision_reason(
    selected_aug: str,
    results: dict[str, dict[str, float]],
    baseline: dict[str, float],
    selection_metric: str = "accuracy",
) -> str:
    if selected_aug == "none":
        return "No candidate improved selected metric vs baseline."
    selected = results.get(selected_aug, {})
    candidate_val = float(selected.get(selection_metric, 0.0))
    baseline_val = float(baseline.get(selection_metric, 0.0))
    delta = candidate_val - baseline_val
    return f"Selected '{selected_aug}' due to {selection_metric} delta {delta:+.4f} vs baseline."
