"""BNNR command-line interface.

Provides fully functional CLI for training, reporting, and dashboard
management without requiring any Python code.

Commands:
- ``bnnr train``: Run BNNR augmentation search
- ``bnnr report``: View/export training reports
- ``bnnr list-augmentations``: List registered augmentations
- ``bnnr list-presets``: List augmentation presets
- ``bnnr list-datasets``: List available datasets
- ``bnnr dashboard serve``: Serve live dashboard
- ``bnnr dashboard export``: Export static dashboard snapshot
- ``bnnr version``: Show version
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import typer

from bnnr.config import load_config
from bnnr.core import BNNRTrainer
from bnnr.dashboard.serve import (
    _find_frontend_dist,
    _get_lan_ip,
    _pick_bind_host,
    _print_qr_code,
    start_dashboard,
)
from bnnr.reporting import load_report

app = typer.Typer(name="bnnr", help="BNNR - Bulletproof Neural Network Recipe", add_completion=False)
dashboard_app = typer.Typer(
    name="dashboard",
    help="Serve or export the new event-based dashboard.",
    add_completion=False,
)
app.add_typer(dashboard_app, name="dashboard")


def _print_pipeline_summary(
    dataset_name: str,
    adapter: Any,
    train_loader: Any,
    val_loader: Any,
    augmentations: list[Any],
    config: Any,
    preset: str,
    custom_data_path: Optional[Path] = None,
) -> None:
    """Print a clear summary of what will be trained, so the user knows exactly what's happening."""
    model = adapter.get_model()
    model_name = model.__class__.__name__
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count samples (handle Subset/Indexed wrappers)
    train_samples: Union[int, str] = "?"
    try:
        train_samples = len(train_loader.dataset)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass
    val_samples: Union[int, str] = "?"
    try:
        val_samples = len(val_loader.dataset)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass

    # Optimizer & scheduler
    opt_name = "?"
    lr: Any = "?"
    for attr in ("optimizer", "_optimizer"):
        opt = getattr(adapter, attr, None)
        if opt is not None:
            opt_name = opt.__class__.__name__
            try:
                lr = opt.param_groups[0]["lr"]
            except (IndexError, KeyError, AttributeError):
                pass
            break
    sched_name = "none"
    for attr in ("scheduler", "_scheduler"):
        sched = getattr(adapter, attr, None)
        if sched is not None:
            sched_name = sched.__class__.__name__
            break

    # AMP
    use_amp = getattr(adapter, "use_amp", False) or getattr(adapter, "_use_amp", False)

    # Augmentations
    aug_names = [getattr(a, "name", a.__class__.__name__) for a in augmentations] if augmentations else ["(none)"]

    typer.echo("")
    typer.echo("=" * 64)
    typer.echo("  BNNR PIPELINE SUMMARY")
    typer.echo("=" * 64)
    typer.echo(f"  Dataset        : {dataset_name}" + (f" ({custom_data_path})" if custom_data_path else ""))
    typer.echo(f"  Model          : {model_name}")
    typer.echo(f"  Parameters     : {total_params:,} total ({trainable_params:,} trainable)")
    typer.echo(f"  Optimizer      : {opt_name} (lr={lr})")
    typer.echo(f"  LR Scheduler   : {sched_name}")
    typer.echo(f"  AMP            : {'enabled' if use_amp else 'disabled'}")
    typer.echo(f"  Device         : {config.device}")
    typer.echo(f"  Train samples  : {train_samples}")
    typer.echo(f"  Val samples    : {val_samples}")
    typer.echo(f"  Batch size     : {train_loader.batch_size}")
    typer.echo("-" * 64)
    typer.echo(f"  Epochs/cand.   : {config.m_epochs}")
    typer.echo(f"  Max iterations : {config.max_iterations}")
    typer.echo(f"  Selection      : {config.selection_metric} ({config.selection_mode})")
    typer.echo(f"  XAI            : {config.xai_method if config.xai_enabled else 'disabled'}")
    typer.echo(f"  Seed           : {config.seed}")
    typer.echo("-" * 64)
    typer.echo(f"  Preset         : {preset}")
    typer.echo(f"  Augmentations  : {len(augmentations)} candidates")
    for name in aug_names:
        typer.echo(f"    • {name}")
    typer.echo("-" * 64)
    typer.echo(f"  Checkpoints    : {config.checkpoint_dir}")
    typer.echo(f"  Reports        : {config.report_dir}")
    typer.echo("=" * 64)

    if dataset_name != "imagefolder":
        typer.echo(
            f"\n  ℹ  Built-in model ({model_name}) is a simple demo CNN."
        )
        typer.echo(
            "     For production, use the Python API with your own model."
        )
        typer.echo(
            "     See: docs/golden_path.md\n"
        )
    else:
        typer.echo(
            f"\n  ℹ  Built-in model ({model_name}) is a generic CNN resized to 64×64."
        )
        typer.echo(
            "     For better results with your own dataset, use the Python API"
        )
        typer.echo(
            "     with a pretrained model (ResNet, EfficientNet, etc.).\n"
        )


@app.command("train")
def train_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    dataset: str = typer.Option(
        "mnist",
        "--dataset",
        help="Dataset: mnist, fashion_mnist, cifar10, stl10, imagefolder, coco_mini, yolo",
    ),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory for dataset download/storage"),
    data_path: Optional[Path] = typer.Option(
        None,
        "--data-path",
        help="Custom data path (imagefolder/coco_mini/yolo)",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for checkpoints and reports"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device: cuda, cpu, auto"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Number of epochs per candidate"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    no_xai: bool = typer.Option(False, "--no-xai", help="Disable XAI generation"),
    augmentation_preset: str = typer.Option(
        "auto",
        "--augmentation-preset",
        "--preset",
        help="Augmentation preset: auto, light, standard, aggressive, gpu",
    ),
    with_dashboard: bool = typer.Option(
        True,
        "--with-dashboard/--without-dashboard",
        help="Enable dashboard: starts server, logs events, opens browser.",
    ),
    dashboard_port: int = typer.Option(8080, "--dashboard-port", help="Dashboard server port"),
    no_auto_open: bool = typer.Option(
        False,
        "--no-auto-open",
        help="Don't auto-open browser when dashboard starts.",
    ),
    dashboard_token: Optional[str] = typer.Option(
        None,
        "--dashboard-token",
        help="Token to protect dashboard control endpoints (pause/resume). "
        "Also configurable via BNNR_DASHBOARD_TOKEN env var.",
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Training batch size"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples", help="Limit training samples"),
    max_val_samples: Optional[int] = typer.Option(None, "--max-val-samples", help="Limit validation samples"),
    num_classes: Optional[int] = typer.Option(None, "--num-classes", help="Number of classes (for imagefolder)"),
) -> None:
    """Run BNNR augmentation search training.

    Examples::

        # Quick MNIST experiment
        bnnr train -c config.yaml --dataset mnist --max-train-samples 1000 -e 2

        # CIFAR-10 with GPU augmentations
        bnnr train -c config.yaml --dataset cifar10 --preset gpu --device cuda

        # Custom ImageFolder dataset
        bnnr train -c config.yaml --dataset imagefolder --data-path /path/to/data
    """
    from bnnr.pipelines import build_pipeline

    cfg = load_config(config)
    # Keep event logs enabled even when live dashboard serving is disabled.
    # B2 CLI flow expects post-run dashboard export to work from the same run dir.
    overrides: dict[str, Any] = {"event_log_enabled": True}
    if output is not None:
        overrides["checkpoint_dir"] = output / "checkpoints"
        overrides["report_dir"] = output / "reports"
    if device is not None:
        overrides["device"] = device
    if epochs is not None:
        overrides["m_epochs"] = epochs
    if seed is not None:
        overrides["seed"] = seed
    if no_xai:
        overrides["xai_enabled"] = False
    cfg = cfg.model_copy(update=overrides)

    try:
        adapter, train_loader, val_loader, augmentations = build_pipeline(
            dataset_name=dataset,
            config=cfg,
            data_dir=data_dir,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            augmentation_preset=augmentation_preset,
            custom_data_path=data_path,
            num_classes=num_classes,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # ── Print pipeline summary so user knows what is being trained ──────
    _print_pipeline_summary(
        dataset_name=dataset,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=augmentations,
        config=cfg,
        preset=augmentation_preset,
        custom_data_path=data_path,
    )

    # ── Start dashboard server in background (before training) ─────────
    dashboard_url = f"http://127.0.0.1:{dashboard_port}/"
    if with_dashboard:
        dashboard_url = start_dashboard(
            run_root=cfg.report_dir,
            port=dashboard_port,
            auto_open=not no_auto_open,
            auth_token=dashboard_token,
        )

    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()

    typer.echo("")
    typer.echo("=" * 64)
    typer.echo("  TRAINING COMPLETE")
    typer.echo("-" * 64)
    typer.echo(f"  Best path      : {result.best_path}")
    typer.echo(f"  Best metrics   : {result.best_metrics}")
    typer.echo(f"  Report JSON    : {result.report_json_path}")
    events_path = result.report_json_path.parent / "events.jsonl"
    if events_path.exists():
        typer.echo(f"  Events (JSONL) : {events_path}")
    if with_dashboard:
        typer.echo(f"  Dashboard      : {dashboard_url}")
    typer.echo("=" * 64)
    typer.echo("")

    if with_dashboard:
        typer.echo("Dashboard is still running — press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            typer.echo("\nShutting down.")


@app.command("report")
def report_command(
    report_path: Path,
    format: str = typer.Option("summary", "--format", "-f", help="Output format: summary, json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """View or export a BNNR training report."""
    report = load_report(report_path)
    fmt = format.lower()
    rendered: str

    if fmt == "summary":
        rendered = "\n".join(
            [
                f"Best path: {report.best_path}",
                f"Best metrics: {report.best_metrics}",
                f"Selected augmentations: {report.selected_augmentations}",
                f"Total checkpoints: {len(report.checkpoints)}",
            ]
        )
    elif fmt == "json":
        payload = {
            "best_path": report.best_path,
            "best_metrics": report.best_metrics,
            "selected_augmentations": report.selected_augmentations,
            "total_time": report.total_time,
        }
        rendered = json.dumps(payload, indent=2)
    elif fmt == "html":
        typer.echo(
            "Error: Legacy HTML report was removed. "
            "Use: bnnr dashboard export --run-dir <run_dir> --out <dir>",
            err=True,
        )
        raise typer.Exit(code=1)
    else:
        typer.echo("Error: Invalid format. Use one of: summary, json, html.", err=True)
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Saved report to: {output}")
        return

    typer.echo(rendered)


def _try_alternate_models(
    dataset_name: str,
    state_dict: dict,
    cfg: Any,
) -> Any:
    """Try alternate model architectures for a known built-in dataset.

    Returns a new ``SimpleTorchAdapter`` with the correct model loaded, or
    ``None`` if no architecture matches.
    """
    import torch as _torch
    import torch.nn as _nn

    from bnnr.adapter import SimpleTorchAdapter

    if dataset_name == "stl10":
        from bnnr.pipelines import _try_stl10_models

        result = _try_stl10_models(state_dict)
        if result is not None:
            model, target_layers = result
            return SimpleTorchAdapter(
                model=model,
                criterion=_nn.CrossEntropyLoss(),
                optimizer=_torch.optim.Adam(model.parameters()),
                target_layers=target_layers,
                device=cfg.device,
            )
    return None


@app.command("analyze")
def analyze_command(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model checkpoint (.pt) or state dict"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to data directory (ImageFolder) or dataset name (e.g. mnist, cifar10)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for analysis_report.json and report.html"),
    task: str = typer.Option("classification", "--task", "-t", help="Task: classification, detection, multilabel"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional YAML config (overrides defaults)"),
    max_worst: int = typer.Option(20, "--max-worst", help="Number of worst predictions to include"),
    no_xai: bool = typer.Option(False, "--no-xai", help="Disable XAI analysis"),
    no_data_quality: bool = typer.Option(False, "--no-data-quality", help="Disable data quality checks"),
    device: Optional[str] = typer.Option(None, "--device", help="Device: cuda, cpu, auto"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for evaluation"),
    output_summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help="Print executive summary and top findings/recommendations to stdout.",
    ),
    cv_folds: int = typer.Option(
        0,
        "--cv-folds",
        help="Optional number of folds for lightweight cross-validation (0 to disable).",
    ),
    xai_samples: int = typer.Option(
        500,
        "--xai-samples",
        help="Number of samples for XAI probe set (more = more accurate, slower).",
    ),
) -> None:
    """Run model analysis: metrics, XAI, data quality, failure patterns, recommendations."""
    from bnnr.analyze import analyze_model

    if not model.exists():
        typer.echo(f"Error: Model path does not exist: {model}", err=True)
        raise typer.Exit(code=1)

    cfg = load_config(config) if config is not None and config.exists() else None
    if cfg is None:
        from bnnr.core import BNNRConfig

        cfg = BNNRConfig(task=task, device=device or "auto", metrics=["accuracy", "f1_macro", "loss"])
    if device is not None:
        cfg = cfg.model_copy(update={"device": device})

    data_path = data.resolve() if data.exists() and data.is_dir() else None
    if data_path is not None:
        dataset_name = "imagefolder"
        data_dir = data_path.parent
    else:
        dataset_name = str(data).lower().strip()
        if dataset_name not in ("mnist", "fashion_mnist", "cifar10", "stl10"):
            typer.echo(
                "Error: --data must be an existing directory (ImageFolder) or one of: mnist, fashion_mnist, cifar10, stl10.",
                err=True,
            )
            raise typer.Exit(code=1)
        data_dir = Path("data")

    try:
        from bnnr.pipelines import build_pipeline

        adapter, _train_loader, val_loader, _augs = build_pipeline(
            dataset_name=dataset_name,
            config=cfg,
            data_dir=data_dir,
            batch_size=batch_size,
            custom_data_path=data_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        import torch

        ckpt = torch.load(model, map_location="cpu", weights_only=False)
    except Exception as exc:
        typer.echo(f"Error loading model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    state = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt["model"]
        if isinstance(ckpt, dict) and "model" in ckpt
        else ckpt
    )
    # If state looks like adapter.state_dict() (has "model" and "optimizer"), use state["model"] for raw weights
    if (
        isinstance(state, dict)
        and "model" in state
        and "optimizer" in state
    ):
        state = state["model"]
    model_obj = getattr(adapter, "model", None)
    if model_obj is not None:
        try:
            model_obj.load_state_dict(state, strict=True)
        except RuntimeError:
            resolved = _try_alternate_models(dataset_name, state, cfg)
            if resolved is not None:
                adapter = resolved
            else:
                typer.echo(
                    "Error: checkpoint does not match any known model architecture for "
                    f"dataset '{dataset_name}'. Use the Python API with your original model class.",
                    err=True,
                )
                raise typer.Exit(code=1)
    else:
        adapter.load_state_dict(state)

    output.mkdir(parents=True, exist_ok=True)
    report = analyze_model(
        adapter,
        val_loader,
        config=cfg,
        task=task,
        output_dir=output,
        run_data_quality=not no_data_quality,
        max_worst=max_worst,
        xai_enabled=not no_xai,
        cv_folds=cv_folds,
        xai_samples=xai_samples,
    )
    report.to_html(output / "report.html")
    if output_summary:
        _print_analyze_summary(report, output)
    else:
        typer.echo(f"Analysis saved to {output}")
        typer.echo(f"  {output / 'analysis_report.json'}")
        typer.echo(f"  {output / 'report.html'}")


def _print_analyze_summary(report: Any, output: Path) -> None:
    """Print concise executive summary and top findings/recommendations for analyze."""
    typer.echo("")
    typer.echo("=" * 64)
    typer.echo("  BNNR ANALYZE SUMMARY")
    typer.echo("=" * 64)
    summary = getattr(report, "executive_summary", {}) or {}
    health = summary.get("health_status", "unknown").upper()
    score = summary.get("health_score")
    if isinstance(score, float):
        score_str = f"{score:.0%}"
    else:
        score_str = "n/a"
    typer.echo(f"  Health      : {health} (score={score_str})")
    if "severity" in summary:
        typer.echo(f"  Severity    : {summary['severity']}")
    acc = report.metrics.get("accuracy", None)
    f1 = report.metrics.get("f1_macro", None)
    if acc is not None or f1 is not None:
        typer.echo("  Metrics     : " + ", ".join(
            part for part in [
                f"accuracy={acc:.1%}" if isinstance(acc, float) else None,
                f"f1_macro={f1:.1%}" if isinstance(f1, float) else None,
            ] if part is not None
        ))
    mean_q = (report.xai_quality_summary or {}).get("mean_quality_score")
    if isinstance(mean_q, float):
        typer.echo(f"  XAI quality : mean_quality_score={mean_q:.3f}")
    key_findings = summary.get("key_findings") or []
    if key_findings:
        typer.echo("-" * 64)
        typer.echo("  Key findings:")
        for f in key_findings[:5]:
            typer.echo(f"    • {f}")
    top_actions = summary.get("top_actions") or []
    if top_actions:
        typer.echo("-" * 64)
        typer.echo("  Top actions:")
        for a in top_actions[:5]:
            typer.echo(f"    • {a}")
    typer.echo("-" * 64)
    typer.echo("  Artifacts:")
    typer.echo(f"    • JSON : {output / 'analysis_report.json'}")
    typer.echo(f"    • HTML : {output / 'report.html'}")
    typer.echo("=" * 64)


@dashboard_app.command("serve")
def dashboard_serve_command(
    run_dir: Path = typer.Option(Path("reports"), "--run-dir"),
    port: int = typer.Option(8080, "--port"),
    frontend_dist: Optional[Path] = typer.Option(None, "--frontend-dist"),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Token to protect control endpoints. Also via BNNR_DASHBOARD_TOKEN env var.",
    ),
) -> None:
    """Serve the BNNR dashboard for live or replay viewing.

    The server binds to 0.0.0.0 so it is reachable from other devices
    on the same network (e.g. your phone).  A QR code is printed so you
    can scan it and open the dashboard instantly.
    """
    try:
        import uvicorn  # noqa: I001
        from bnnr.dashboard.backend import create_dashboard_app, list_runs  # noqa: I001
    except ImportError as exc:  # pragma: no cover - optional dependency
        typer.echo(
            "Error: Dashboard backend dependencies are missing. "
            "Install with: pip install -e '.[dashboard]'",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    from bnnr.dashboard.backend import _normalize_run_root  # noqa: I001

    resolved = _normalize_run_root(run_dir.resolve())
    typer.echo(f"Serving dashboard API for run dir: {resolved}")
    available = list_runs(resolved)
    typer.echo(f"Available runs: {len(available)}")

    local_url = f"http://127.0.0.1:{port}/"
    lan_ip = _get_lan_ip()
    lan_url = f"http://{lan_ip}:{port}/"

    typer.echo(f"  Local URL    : {local_url}")
    typer.echo(f"  Network URL  : {lan_url}")
    _print_qr_code(lan_url)

    # If user did not pass --frontend-dist, auto-discover the newest built dist.
    effective_frontend_dist = frontend_dist
    if effective_frontend_dist is None:
        effective_frontend_dist = _find_frontend_dist(auto_build=False)
        if effective_frontend_dist is not None:
            typer.echo(f"  Frontend dist  : {effective_frontend_dist} (auto-detected)")
        else:
            typer.echo("  Frontend dist  : not found — run: cd dashboard_web && npm run build")
    elif not (effective_frontend_dist / "index.html").exists():
        typer.echo(f"  Warning: --frontend-dist has no index.html: {effective_frontend_dist}")

    bind_host = _pick_bind_host(port)
    if bind_host != "0.0.0.0":
        typer.echo(
            f"  Bind host      : {bind_host} (fallback from 0.0.0.0)",
        )
    app_instance = create_dashboard_app(resolved, static_dir=effective_frontend_dist, auth_token=token, mode="serve")
    uvicorn.run(
        app_instance,
        host=bind_host,  # noqa: S104
        port=port,
        log_level="info",
    )


@dashboard_app.command("export")
def dashboard_export_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    out: Path = typer.Option(..., "--out"),
    frontend_dist: Optional[Path] = typer.Option(None, "--frontend-dist"),
) -> None:
    """Export a standalone dashboard snapshot to a directory."""
    from bnnr.dashboard.exporter import export_dashboard_snapshot

    exported = export_dashboard_snapshot(
        run_dir=run_dir,
        out_dir=out,
        frontend_dist=frontend_dist,
    )
    typer.echo(f"Exported dashboard snapshot to: {exported}")


@app.command("list-augmentations")
def list_augmentations(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """List all registered augmentations."""
    from bnnr.augmentations import AugmentationRegistry

    # Also trigger kornia registrations
    try:
        import bnnr.kornia_aug  # noqa: F401
    except ImportError:
        pass

    names = AugmentationRegistry.list_all()
    if not names:
        typer.echo("No augmentations registered.")
        return
    for name in names:
        typer.echo(name)
        if verbose:
            cls = AugmentationRegistry.get(name)
            doc = (cls.__doc__ or "").strip().split("\n")[0] or "No description"
            gpu_tag = " [GPU]" if getattr(cls, "device_compatible", False) else " [CPU]"
            typer.echo(f"  - class: {cls.__name__}{gpu_tag}")
            typer.echo(f"  - doc: {doc}")


@app.command("list-presets")
def list_presets_command() -> None:
    """List available augmentation presets."""
    from bnnr.presets import list_presets

    presets = list_presets()
    typer.echo("Available augmentation presets:")
    typer.echo(f"  {'auto':<15} Auto-select best augmentations for current hardware")
    for name, desc in presets.items():
        typer.echo(f"  {name:<15} {desc}")


@app.command("list-datasets")
def list_datasets_command() -> None:
    """List available built-in datasets."""
    from bnnr.pipelines import list_datasets

    ds = list_datasets()
    typer.echo("Available datasets:")
    for name in ds:
        typer.echo(f"  {name}")


@app.command("version")
def version_command() -> None:
    """Show BNNR version."""
    from bnnr import __version__

    typer.echo(f"bnnr version {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
