"""Tests for report serialization and loading."""

from __future__ import annotations

from datetime import datetime

from bnnr.core import BNNRConfig
from bnnr.reporting import CheckpointInfo, Reporter, compare_runs, load_report


def test_checkpoint_info_dataclass(temp_dir) -> None:
    cp = CheckpointInfo(
        iteration=0,
        augmentation="baseline",
        epoch=1,
        metrics={"accuracy": 0.5},
        checkpoint_path=temp_dir / "cp.pt",
        xai_paths=[],
        preview_pairs=[],
        probe_labels=[],
        active_path="baseline",
        timestamp=datetime.now(),
    )
    assert cp.augmentation == "baseline"


def test_reporter_finalize_creates_json(temp_dir) -> None:
    reporter = Reporter(report_dir=temp_dir)
    cfg = BNNRConfig(checkpoint_dir=temp_dir / "c", report_dir=temp_dir)
    reporter.start(cfg)
    reporter.log_checkpoint(0, "baseline", 1, {"accuracy": 0.5}, temp_dir / "a.pt")
    result = reporter.finalize(best_path="baseline", best_metrics={"accuracy": 0.5}, selected_augmentations=[])
    assert result.report_json_path.exists()
    assert result.report_html_path is None
    assert (reporter.run_dir / "events.jsonl").exists()


def test_reporter_keeps_iteration_summaries(temp_dir) -> None:
    reporter = Reporter(report_dir=temp_dir)
    cfg = BNNRConfig(checkpoint_dir=temp_dir / "c", report_dir=temp_dir, xai_enabled=False)
    reporter.start(cfg)
    reporter.log_checkpoint(0, "baseline", 1, {"accuracy": 0.5, "loss": 1.0}, temp_dir / "a.pt")
    reporter.log_iteration_summary(
        1,
        {"augmentation_1": {"accuracy": 0.6, "loss": 0.9}},
        "augmentation_1",
        baseline_metrics={"accuracy": 0.5, "loss": 1.0},
        top_candidates=["augmentation_1"],
    )
    _ = reporter.finalize(
        best_path="augmentation_1",
        best_metrics={"accuracy": 0.6},
        selected_augmentations=["augmentation_1"],
        analysis={"xai_insights": ["insight-1"]},
    )
    report_payload = reporter.run_dir / "report.json"
    assert report_payload.exists()
    data = report_payload.read_text(encoding="utf-8")
    assert "iteration_summaries" in data


def test_load_report_and_compare(temp_dir) -> None:
    reporter = Reporter(report_dir=temp_dir)
    cfg = BNNRConfig(checkpoint_dir=temp_dir / "c", report_dir=temp_dir)
    reporter.start(cfg)
    reporter.log_checkpoint(0, "baseline", 1, {"accuracy": 0.5}, temp_dir / "a.pt")
    result = reporter.finalize(best_path="baseline", best_metrics={"accuracy": 0.5}, selected_augmentations=[])

    loaded = load_report(result.report_json_path)
    comp = compare_runs([loaded], metrics=["accuracy"])
    assert "run_0" in comp


def test_reporter_can_disable_event_log(temp_dir) -> None:
    reporter = Reporter(report_dir=temp_dir)
    cfg = BNNRConfig(
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir,
        event_log_enabled=False,
    )
    reporter.start(cfg)
    reporter.log_checkpoint(0, "baseline", 1, {"accuracy": 0.5}, temp_dir / "a.pt")
    _ = reporter.finalize(best_path="baseline", best_metrics={"accuracy": 0.5}, selected_augmentations=[])
    assert not (reporter.run_dir / "events.jsonl").exists()
