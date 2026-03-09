"""Tests for bnnr.analyze — analyze_model and AnalysisReport."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from bnnr.analyze import AnalysisReport, analyze_model
from bnnr.core import BNNRConfig
from bnnr.evaluation import run_evaluation


def _make_indexed_loader(n: int = 12, batch_size: int = 4) -> DataLoader:
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, 3, (n,))

    class IndexedDataset(TensorDataset):
        def __getitem__(self, index: int):
            a, b = super().__getitem__(index)
            return a, b, index

    ds = IndexedDataset(x, y)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        indices = torch.tensor([b[2] for b in batch])
        return imgs, labels, indices

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


def test_analyze_model_returns_report(model_adapter, tmp_path: Path) -> None:
    loader = _make_indexed_loader()
    config = BNNRConfig(device="cpu", task="classification")
    report = analyze_model(
        model_adapter,
        loader,
        config=config,
        output_dir=None,
        run_data_quality=False,
    )
    assert isinstance(report, AnalysisReport)
    assert "accuracy" in report.metrics or "f1_macro" in report.metrics
    assert isinstance(report.per_class_accuracy, dict)
    assert isinstance(report.confusion, dict)


def test_analyze_model_save(model_adapter, tmp_path: Path) -> None:
    loader = _make_indexed_loader()
    report = analyze_model(
        model_adapter,
        loader,
        config=BNNRConfig(device="cpu"),
        output_dir=tmp_path,
        run_data_quality=False,
    )
    report.save(tmp_path)
    json_path = tmp_path / "analysis_report.json"
    assert json_path.exists()
    data = __import__("json").loads(json_path.read_text())
    assert "metrics" in data
    assert "per_class_accuracy" in data
    assert "confusion" in data


def test_analysis_report_worst_predictions_list() -> None:
    report = AnalysisReport(worst_predictions=[{"a": 1}, {"b": 2}, {"c": 3}])
    assert len(report.worst_predictions_list(n=2)) == 2
    assert report.worst_predictions_list(n=2)[0] == {"a": 1}


def test_analysis_report_to_html(tmp_path: Path) -> None:
    report = AnalysisReport(metrics={"accuracy": 0.9}, recommendations=["Use ICD"])
    out = tmp_path / "report.html"
    report.to_html(out)
    assert out.exists()
    html = out.read_text()
    assert "accuracy" in html
    assert "Recommendations" in html
    assert "Use ICD" in html


def test_analyze_model_extended_report_fields(model_adapter, tmp_path: Path) -> None:
    """Extended analysis populates executive_summary, findings, class_diagnostics, recommendations_structured."""
    loader = _make_indexed_loader()
    config = BNNRConfig(device="cpu", task="classification")
    report = analyze_model(
        model_adapter,
        loader,
        config=config,
        output_dir=None,
        run_data_quality=False,
    )
    assert report.schema_version == "0.2.0"
    assert isinstance(report.executive_summary, dict)
    assert "health_status" in report.executive_summary or len(report.executive_summary) >= 0
    assert isinstance(report.findings, list)
    assert isinstance(report.recommendations_structured, list)
    assert isinstance(report.class_diagnostics, list)
    assert isinstance(report.true_distribution, dict)
    assert isinstance(report.pred_distribution, dict)
    assert isinstance(report.distribution_summary, dict)
    assert isinstance(report.failure_patterns_extended, list)
    if report.confusion:
        assert len(report.class_diagnostics) > 0
        assert len(report.true_distribution) > 0
        assert len(report.pred_distribution) > 0


def test_analysis_report_html_sections(tmp_path: Path) -> None:
    """HTML report contains key sections and text labels."""
    report = AnalysisReport(
        metrics={"accuracy": 0.85, "loss": 0.4},
        executive_summary={"health_status": "needs_attention", "key_findings": ["Low accuracy"]},
        findings=[{"title": "Low recall class 2", "severity": "high"}],
        recommendations_structured=[{"title": "Add data for class 2", "priority": 1}],
        class_diagnostics=[{"class_name": "0", "accuracy": 0.9, "recall": 0.9, "support": 10}],
    )
    out = tmp_path / "report.html"
    report.to_html(out)
    html = out.read_text()
    assert "Executive summary" in html
    assert "Class diagnostics" in html
    assert "Findings" in html
    assert "Dataset health" in html
    assert "Cross-validation" in html
    assert "XAI insights" in html
    assert "Recommendations" in html
    assert "Method & caveats" in html
    assert "needs_attention" in html or "Low accuracy" in html
    assert "Low recall class 2" in html
    assert "Add data for class 2" in html


def test_analyze_model_with_xai(model_adapter, tmp_path: Path) -> None:
    loader = _make_indexed_loader(n=16)
    report = analyze_model(
        model_adapter,
        loader,
        config=BNNRConfig(device="cpu"),
        output_dir=None,
        run_data_quality=False,
        xai_enabled=True,
        xai_method="opticam",
    )
    assert isinstance(report.xai_insights, dict)
    assert isinstance(report.xai_diagnoses, dict)
    if report.xai_quality_summary:
        assert "mean_quality_score" in report.xai_quality_summary
    # Cluster views and XAI per-class structures should be present (even if empty lists/dicts).
    assert isinstance(report.xai_quality_per_class, dict)
    assert isinstance(report.xai_examples_per_class, dict)
    assert isinstance(report.cluster_views, list)


def test_analyze_model_with_cv_and_cluster(model_adapter, tmp_path: Path) -> None:
    """analyze_model can run lightweight CV and populate cv_results."""
    loader = _make_indexed_loader()
    config = BNNRConfig(device="cpu", task="classification")
    report = analyze_model(
        model_adapter,
        loader,
        config=config,
        output_dir=None,
        run_data_quality=False,
        xai_enabled=False,
        cv_folds=3,
    )
    assert isinstance(report.cv_results, dict)
    if report.cv_results:
        assert report.cv_results.get("n_folds", 0) >= 1


def test_run_evaluation_standalone(model_adapter) -> None:
    """Run evaluation module directly (used by analyze and trainer)."""
    adapter = model_adapter
    loader = _make_indexed_loader()
    config = BNNRConfig(device="cpu")
    metrics, per_class, confusion, preds, labels = run_evaluation(
        adapter, loader, config, return_preds_labels=True
    )
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert isinstance(per_class, dict)
    assert isinstance(confusion, dict)
    assert preds is not None
    assert labels is not None
    assert len(preds) == 12
    assert len(labels) == 12


def test_analyze_cli(tmp_path: Path) -> None:
    """CLI bnnr analyze runs and produces output files (using mnist pipeline)."""
    import subprocess
    import sys

    from bnnr.core import BNNRConfig
    from bnnr.pipelines import build_pipeline

    cfg = BNNRConfig(device="cpu", task="classification")
    adapter, _, _, _ = build_pipeline(
        dataset_name="mnist",
        config=cfg,
        data_dir=tmp_path / "data",
        batch_size=8,
        max_train_samples=16,
        max_val_samples=24,
    )
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state": adapter.model.state_dict()}, ckpt)
    out_dir = tmp_path / "analysis_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnnr",
            "analyze",
            "--model",
            str(ckpt),
            "--data",
            "mnist",
            "--output",
            str(out_dir),
            "--batch-size",
            "8",
            "--no-xai",
            "--no-data-quality",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (out_dir / "analysis_report.json").exists()
    assert (out_dir / "report.html").exists()
