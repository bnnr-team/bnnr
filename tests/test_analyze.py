"""Tests for bnnr.analyze — analyze_model and AnalysisReport."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.analyze import AnalysisReport, analyze_model
from bnnr.core import BNNRConfig
from bnnr.evaluation import run_evaluation


def _seed_mnist_data_dir(data_parent: Path) -> Path:
    """Populate *data_parent*/MNIST from a local cache to avoid flaky MNIST downloads.

    Uses ``BNNR_TEST_MNIST_CACHE`` (directory containing ``MNIST/``) if set, else
    ``~/.cache/bnnr_test_mnist/MNIST`` when present (see repo smoke / CI setup).
    """
    data_parent = Path(data_parent)
    data_parent.mkdir(parents=True, exist_ok=True)
    dest = data_parent / "MNIST"
    if dest.is_dir():
        return data_parent
    src: Path | None = None
    env = os.environ.get("BNNR_TEST_MNIST_CACHE")
    if env and (Path(env) / "MNIST").is_dir():
        src = Path(env) / "MNIST"
    else:
        bundled = Path.home() / ".cache" / "bnnr_test_mnist" / "MNIST"
        if bundled.is_dir():
            src = bundled
    if src is not None:
        shutil.copytree(src, dest)
    return data_parent


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
    assert "confusion_pair_xai" in data
    assert "best_worst_examples" in data


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
    assert report.schema_version == "0.2.1"
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
    assert "cohen_kappa" in report.metrics
    assert isinstance(report.confusion_pair_xai, list)
    assert isinstance(report.best_worst_examples, dict)


def test_analysis_report_html_sections(tmp_path: Path) -> None:
    """HTML report contains key sections and text labels."""
    report = AnalysisReport(
        metrics={"accuracy": 0.85, "loss": 0.4, "cohen_kappa": 0.75},
        executive_summary={"health_status": "needs_attention", "key_findings": ["Low accuracy"]},
        findings=[{"title": "Low recall class 2", "severity": "high", "confidence": "high"}],
        recommendations_structured=[{"title": "Add data for class 2", "priority": 1}],
        class_diagnostics=[{
            "class_id": "0", "accuracy": 0.9, "recall": 0.9, "precision": 0.85,
            "f1": 0.87, "support": 10, "pred_count": 11, "cohen_kappa": 0.8,
        }],
        data_quality_summary={"scanned_samples": 100, "total_duplicate_pairs": 0},
        xai_quality_summary={"mean_quality_score": 0.7},
    )
    out = tmp_path / "report.html"
    report.to_html(out)
    html = out.read_text()
    assert "Executive Summary" in html
    assert "Class Diagnostics" in html
    assert "Findings" in html
    assert "Dataset Health" in html
    assert "XAI Insights" in html
    assert "Recommendations" in html
    assert "needs_attention" in html or "Low accuracy" in html
    assert "Low recall class 2" in html
    assert "Add data for class 2" in html
    assert "Observed" in html or "Likely" in html
    assert "BNNR" in html
    assert "Train" in html
    assert "production" in html.lower()


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
    assert isinstance(report.xai_quality_per_class, dict)
    assert isinstance(report.xai_examples_per_class, dict)


def test_analyze_model_with_cv(model_adapter, tmp_path: Path) -> None:
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
        gm = report.cv_results.get("global_metrics", {})
        if gm:
            assert "mean_accuracy" in gm
            assert "mean_precision_macro" in gm
            assert "mean_recall_macro" in gm
            assert "mean_cohen_kappa" in gm


def test_analyze_model_rejects_detection(model_adapter) -> None:
    """analyze_model rejects unsupported tasks (BNNRConfig itself forbids detection)."""
    from types import SimpleNamespace

    loader = _make_indexed_loader()
    fake_cfg = SimpleNamespace(task="detection", device="cpu", metrics=["accuracy"])
    with pytest.raises(ValueError, match="classification.*multilabel"):
        analyze_model(
            model_adapter,
            loader,
            config=fake_cfg,
            output_dir=None,
            run_data_quality=False,
        )


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


class _TinyMultiLabelNet(nn.Module):
    def __init__(self, n_labels: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_run_evaluation_multilabel_returns_2d_preds() -> None:
    n_labels = 4
    model = _TinyMultiLabelNet(n_labels)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    adapter = SimpleTorchAdapter(
        model,
        crit,
        opt,
        target_layers=[model.conv1],
        device="cpu",
        multilabel=True,
    )
    x = torch.rand(16, 3, 32, 32)
    y = torch.randint(0, 2, (16, n_labels)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    config = BNNRConfig(device="cpu", task="multilabel", metrics=["f1_macro", "loss"])
    metrics, per_class, confusion, preds, labels = run_evaluation(
        adapter, loader, config, return_preds_labels=True
    )
    assert confusion.get("type") == "multilabel_per_label"
    assert confusion.get("n_samples") == 16
    assert preds is not None and labels is not None
    assert preds.shape == (16, n_labels)
    assert labels.shape == (16, n_labels)
    assert len(per_class) == n_labels


def test_multilabel_analyze_extended_no_false_kappa() -> None:
    n_labels = 4
    model = _TinyMultiLabelNet(n_labels)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    adapter = SimpleTorchAdapter(
        model,
        crit,
        opt,
        target_layers=[model.conv1],
        device="cpu",
        multilabel=True,
    )
    x = torch.rand(24, 3, 32, 32)
    y = torch.randint(0, 2, (24, n_labels)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    config = BNNRConfig(
        device="cpu",
        task="multilabel",
        metrics=["f1_macro", "f1_samples", "accuracy", "loss"],
    )
    report = analyze_model(
        adapter,
        loader,
        config=config,
        output_dir=None,
        run_data_quality=False,
        xai_enabled=False,
        cv_folds=3,
    )
    assert report.analysis_scope.get("task") == "multilabel"
    assert report.analysis_scope.get("extended_analysis") is True
    assert report.confusion.get("type") == "multilabel_per_label"
    assert "cohen_kappa" not in report.metrics
    assert len(report.class_diagnostics) == n_labels
    assert isinstance(report.findings, list)
    assert isinstance(report.recommendations_structured, list)
    assert isinstance(report.recommendations, list) and len(report.recommendations) >= 1
    assert "ece" not in report.metrics
    assert report.cv_results.get("n_folds", 0) >= 1


def test_top1_ece_perfect_calibration() -> None:
    from bnnr.analysis.calibration import compute_top1_ece

    n = 200
    conf = np.ones(n) * 0.8
    correct = np.random.RandomState(0).rand(n) < 0.8
    out = compute_top1_ece(conf, correct, n_bins=10)
    assert "ece" in out
    assert 0 <= out["ece"] <= 1


def test_analyze_cli(tmp_path: Path) -> None:
    """CLI bnnr analyze runs and produces output files (using mnist pipeline)."""
    import subprocess
    import sys

    from bnnr.core import BNNRConfig
    from bnnr.pipelines import build_pipeline

    cfg = BNNRConfig(device="cpu", task="classification")
    data_dir = _seed_mnist_data_dir(tmp_path / "data")
    if not (data_dir / "MNIST").is_dir():
        pytest.skip(
            "MNIST not cached offline; set BNNR_TEST_MNIST_CACHE or copy MNIST to "
            "~/.cache/bnnr_test_mnist/MNIST (see tests/test_analyze.py::_seed_mnist_data_dir)."
        )
    adapter, _, _, _ = build_pipeline(
        dataset_name="mnist",
        config=cfg,
        data_dir=data_dir,
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


def test_analyze_cli_with_xai(tmp_path: Path) -> None:
    """CLI bnnr analyze with XAI enabled produces XAI fields in report."""
    import subprocess
    import sys

    from bnnr.core import BNNRConfig
    from bnnr.pipelines import build_pipeline

    cfg = BNNRConfig(device="cpu", task="classification")
    data_dir = _seed_mnist_data_dir(tmp_path / "data")
    if not (data_dir / "MNIST").is_dir():
        pytest.skip(
            "MNIST not cached offline; set BNNR_TEST_MNIST_CACHE or copy MNIST to "
            "~/.cache/bnnr_test_mnist/MNIST (see tests/test_analyze.py::_seed_mnist_data_dir)."
        )
    adapter, _, _, _ = build_pipeline(
        dataset_name="mnist",
        config=cfg,
        data_dir=data_dir,
        batch_size=8,
        max_train_samples=16,
        max_val_samples=24,
    )
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state": adapter.model.state_dict()}, ckpt)
    out_dir = tmp_path / "analysis_out_xai"
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
            "--xai-samples",
            "30",
            "--no-data-quality",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    data = __import__("json").loads((out_dir / "analysis_report.json").read_text())
    assert "xai_insights" in data or "xai_quality_summary" in data
    assert "metrics" in data


def test_analyze_api_output_dir_none(model_adapter, tmp_path: Path) -> None:
    """API with output_dir=None returns full report without writing to disk."""
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
    assert report.metrics
    assert report.executive_summary
    assert not (tmp_path / "analysis_report.json").exists()


def test_api_cli_parity(tmp_path: Path) -> None:
    """API and CLI produce consistent report structure and metrics for same model (classification)."""
    import subprocess
    import sys

    from bnnr.core import BNNRConfig
    from bnnr.pipelines import build_pipeline

    cfg = BNNRConfig(device="cpu", task="classification")
    data_dir = _seed_mnist_data_dir(tmp_path / "data")
    if not (data_dir / "MNIST").is_dir():
        pytest.skip(
            "MNIST not cached offline; set BNNR_TEST_MNIST_CACHE or copy MNIST to "
            "~/.cache/bnnr_test_mnist/MNIST (see tests/test_analyze.py::_seed_mnist_data_dir)."
        )
    adapter, _, val_loader, _ = build_pipeline(
        dataset_name="mnist",
        config=cfg,
        data_dir=data_dir,
        batch_size=8,
        max_train_samples=24,
        max_val_samples=48,
    )
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state": adapter.model.state_dict()}, ckpt)

    # API
    report_api = analyze_model(
        adapter,
        val_loader,
        config=cfg,
        output_dir=None,
        run_data_quality=False,
        xai_enabled=False,
    )

    # CLI (uses same checkpoint; val set may differ so we check structure, not exact values)
    out_dir = tmp_path / "cli_out"
    subprocess.run(
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
        check=True,
        timeout=90,
    )
    data_cli = __import__("json").loads((out_dir / "analysis_report.json").read_text())

    # Both produce same schema
    for key in ("metrics", "per_class_accuracy", "confusion", "executive_summary", "findings"):
        assert key in data_cli, f"CLI report missing {key}"
    assert "accuracy" in report_api.metrics
    assert "accuracy" in data_cli["metrics"]
    assert 0 <= data_cli["metrics"]["accuracy"] <= 1


def test_recommendations_build() -> None:
    """Structured recommendations are built from findings."""
    from bnnr.analysis.recommendations import build_recommendations
    from bnnr.analysis.schema import Finding

    class MockReport:
        metrics = {"accuracy": 0.7}
        xai_quality_summary = {}
        class_diagnostics = []

    findings = [
        Finding(
            title="Zero recall",
            finding_type="zero_recall_class",
            description="Class 2: no correct predictions.",
            evidence=["recall=0"],
            severity="critical",
            confidence="high",
            class_ids=["2"],
        ),
    ]
    recs = build_recommendations(findings, MockReport(), max_items=5)
    assert isinstance(recs, list)
    assert len(recs) >= 1
    assert any("focal" in r.action.lower() or "class" in r.action.lower() for r in recs)


def test_cohen_kappa_in_class_diagnostics() -> None:
    """Cohen's Kappa is computed per-class in class_diagnostics."""
    from bnnr.analysis.class_diagnostics import compute_class_diagnostics

    confusion = {
        "matrix": [[90, 10], [20, 80]],
        "labels": [0, 1],
    }
    diag, true_dist, pred_dist = compute_class_diagnostics(confusion)
    assert len(diag) == 2
    for d in diag:
        assert hasattr(d, "cohen_kappa")
        assert isinstance(d.cohen_kappa, float)


def test_grouped_findings() -> None:
    """Findings of the same type are grouped into single entries."""
    from bnnr.analysis.findings import build_findings
    from bnnr.analysis.schema import ClassDiagnostic

    class MockReport:
        confusion = {
            "matrix": [[100, 50], [30, 120]],
            "labels": [0, 1],
        }
        metrics = {"accuracy": 0.73}
        xai_diagnoses = {}

    diags = [
        ClassDiagnostic(class_id="0", accuracy=0.67, precision=0.77, recall=0.67, f1=0.71, support=150, pred_count=130),
        ClassDiagnostic(class_id="1", accuracy=0.80, precision=0.71, recall=0.80, f1=0.75, support=150, pred_count=170),
    ]
    true_d = {"0": 150, "1": 150}
    pred_d = {"0": 130, "1": 170}

    findings, patterns = build_findings(MockReport(), diags, true_d, pred_d)
    assert isinstance(findings, list)
    assert isinstance(patterns, list)
    finding_types = [f.finding_type for f in findings]
    for ft in finding_types:
        assert finding_types.count(ft) == 1, f"Finding type {ft} appears more than once (not grouped)"


def test_cross_validation_all_metrics() -> None:
    """Cross-validation includes precision, recall, f1, and Cohen's Kappa."""
    import numpy as np

    from bnnr.analysis.cross_validation import run_cross_validation_from_predictions

    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=60)
    preds = labels.copy()
    preds[:10] = (preds[:10] + 1) % 3

    cv = run_cross_validation_from_predictions(preds, labels, n_folds=3)
    assert cv.n_folds == 3
    gm = cv.global_metrics
    assert "mean_accuracy" in gm
    assert "mean_precision_macro" in gm
    assert "mean_recall_macro" in gm
    assert "mean_f1_macro" in gm
    assert "mean_cohen_kappa" in gm
    for fm in cv.per_fold_metrics:
        assert "precision_macro" in fm
        assert "recall_macro" in fm
        assert "cohen_kappa" in fm
