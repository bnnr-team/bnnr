"""
YOLO COCO128 detection showcase for BNNR with live dashboard support.

This example is focused on lightweight, practical validation:
- dataset: YOLO format (COCO128 by default)
- model: Faster R-CNN pipeline via BNNR YOLO loader
- metrics: mAP@0.5, mAP@[.5:.95]
- augmentations: bbox-aware geometric + DetectionICD/DetectionAICD

Quick test:
    PYTHONPATH=src python examples/detection/showcase_yolo_coco128.py \
        --data-path data/coco128/data.yaml --with-dashboard --quick

COCO128 is fetched automatically when ``.../coco128/data.yaml`` is missing (see ``--no-auto-download``).

Without dashboard:
    PYTHONPATH=src python examples/detection/showcase_yolo_coco128.py \
        --data-path data/coco128/data.yaml --without-dashboard --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from bnnr import BNNRConfig, BNNRTrainer, start_dashboard
from bnnr.config import load_config
from bnnr.example_data import resolve_yolo_example_data_yaml
from bnnr.detection_augmentations import (
    DetectionHorizontalFlip,
    DetectionMixUp,
    DetectionRandomRotate90,
    DetectionRandomScale,
    DetectionVerticalFlip,
    MosaicAugmentation,
)
from bnnr.detection_icd import DetectionAICD, DetectionICD
from bnnr.pipelines import build_pipeline
from bnnr.reporting import Reporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "BNNR YOLO COCO128 detection showcase — bbox-aware augmentations, "
            "DetectionICD/AICD, live dashboard."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/detection/yolo_coco128_example.yaml"),
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/coco128/data.yaml"),
        help="Path to YOLO data.yaml (default: data/coco128/data.yaml).",
    )
    p.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Do not download COCO128 when .../coco128/data.yaml is missing.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--m-epochs",
        type=int,
        default=3,
        help="Epochs per branch per decision round.",
    )
    p.add_argument(
        "--decisions",
        type=int,
        default=3,
        help="Number of decision rounds.",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training sample cap.",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional validation sample cap.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: smaller caps + shorter run.",
    )

    dash = p.add_mutually_exclusive_group()
    dash.add_argument("--with-dashboard", dest="with_dashboard", action="store_true")
    dash.add_argument("--without-dashboard", dest="with_dashboard", action="store_false")
    p.set_defaults(with_dashboard=True)
    p.add_argument("--dashboard-port", type=int, default=8080)

    auto = p.add_mutually_exclusive_group()
    auto.add_argument("--dashboard-auto-open", dest="dashboard_auto_open", action="store_true")
    auto.add_argument("--no-dashboard-auto-open", dest="dashboard_auto_open", action="store_false")
    p.set_defaults(dashboard_auto_open=True)
    p.add_argument(
        "--dashboard-build-frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def _load_yolo_data_spec(yaml_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YOLO showcase. Install with: pip install pyyaml"
        ) from exc

    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(
            f"Invalid YOLO data file: {yaml_path}. Expected a YAML mapping with at least 'train' and 'val'."
        )

    if "train" not in spec or "val" not in spec:
        bnnr_like_keys = {"task", "m_epochs", "max_iterations", "selection_metric", "report_dir"}
        if bnnr_like_keys.intersection(set(spec.keys())):
            raise ValueError(
                f"Invalid --data-path: {yaml_path} looks like a BNNR config YAML, not a YOLO data.yaml.\n"
                "Use --config for BNNR config and --data-path for dataset spec.\n"
                "Example: --config examples/configs/detection/yolo_coco128_example.yaml "
                "--data-path data/coco128/data.yaml"
            )
        raise ValueError(
            f"Invalid YOLO data.yaml: {yaml_path}. Required keys: 'train' and 'val'."
        )

    return spec


def _read_yolo_class_names(spec: dict[str, Any]) -> list[str]:
    if not isinstance(spec, dict):
        return []

    names = spec.get("names")
    if isinstance(names, list):
        return [str(v) for v in names]
    if isinstance(names, dict):
        def _k(raw: Any) -> tuple[int, str]:
            try:
                return (0, str(int(raw)))
            except (TypeError, ValueError):
                return (1, str(raw))

        ordered = [names[k] for k in sorted(names.keys(), key=_k)]
        return [str(v) for v in ordered]

    nc = spec.get("nc")
    if isinstance(nc, int) and nc > 0:
        return [f"class_{i}" for i in range(nc)]
    return []


def _build_detection_augmentations(seed: int, quick: bool = False) -> list:
    """Detection-focused candidate pool aligned with BNNR's core idea."""
    augs: list = [
        DetectionHorizontalFlip(probability=0.5, name_override="det_hflip", random_state=seed),
        DetectionVerticalFlip(probability=0.5, name_override="det_vflip", random_state=seed + 1),
        DetectionRandomRotate90(probability=0.5, name_override="det_rotate90", random_state=seed + 2),
        DetectionRandomScale(
            scale_range=(0.85, 1.15), probability=0.5, name_override="det_scale", random_state=seed + 3,
        ),
        DetectionICD(probability=0.5, name_override="det_icd", random_state=seed + 4),
        DetectionAICD(probability=0.5, name_override="det_aicd", random_state=seed + 5),
    ]
    if not quick:
        augs.extend([
            MosaicAugmentation(
                output_size=(480, 480),
                probability=0.4,
                name_override="det_mosaic",
                random_state=seed + 6,
            ),
            DetectionMixUp(
                alpha_range=(0.3, 0.7),
                probability=0.4,
                name_override="det_mixup",
                random_state=seed + 7,
            ),
        ])
    return augs


def main() -> None:
    args = parse_args()

    try:
        data_yaml = resolve_yolo_example_data_yaml(
            args.data_path,
            auto_download=not args.no_auto_download,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        print(f"[error] {exc}")
        raise SystemExit(2) from exc

    if not data_yaml.is_file():
        print(f"[error] --data-path must point to YOLO data.yaml file: {data_yaml}")
        raise SystemExit(2)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = BNNRConfig(task="detection")

    if args.quick:
        args.max_train_samples = args.max_train_samples or 128
        args.max_val_samples = args.max_val_samples or 64
        args.m_epochs = 1
        args.decisions = 1

    try:
        data_spec = _load_yolo_data_spec(data_yaml)
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(2)

    class_names = _read_yolo_class_names(data_spec)
    names_with_bg = ["background", *class_names] if class_names else None

    if config.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = config.device

    overrides: dict[str, object] = {
        "task": "detection",
        "m_epochs": args.m_epochs,
        "max_iterations": args.decisions,
        "event_log_enabled": args.with_dashboard,
        "device": resolved_device,
    }
    if names_with_bg is not None:
        overrides["detection_class_names"] = names_with_bg
    if args.quick:
        overrides["candidate_pruning_enabled"] = True
        overrides["candidate_pruning_relative_threshold"] = 0.7
        overrides["candidate_pruning_warmup_epochs"] = 1
    config = config.model_copy(update=overrides)

    total_epochs = args.m_epochs * (args.decisions + 1)
    print()
    print("=" * 68)
    print("  BNNR  ·  YOLO COCO128  Detection Showcase")
    print("-" * 68)
    print(f"  Dataset path            : {data_yaml}")
    print(f"  Max main-path epochs    : ~{total_epochs}")
    print(f"  Decision rounds         : {args.decisions}")
    print(f"  Epochs per branch       : {args.m_epochs}")
    print(f"  Device                  : {config.device}")
    print(f"  Class count (no bg)     : {len(class_names) if class_names else 'from runtime labels'}")
    print(f"  Dashboard               : {'enabled' if args.with_dashboard else 'disabled'}")
    if args.quick:
        print("  Mode                    : QUICK TEST")
    print("=" * 68)
    print()

    adapter, train_loader, val_loader, _ = build_pipeline(
        "yolo",
        config=config,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        augmentation_preset="none",
        custom_data_path=data_yaml,
        num_classes=len(names_with_bg) if names_with_bg else None,
    )
    augmentations = _build_detection_augmentations(seed=config.seed, quick=args.quick)

    dashboard_url = ""
    if args.with_dashboard:
        dashboard_url = start_dashboard(
            config.report_dir,
            port=args.dashboard_port,
            auto_open=args.dashboard_auto_open,
            build_frontend=args.dashboard_build_frontend,
        )

    train_count = len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1
    val_count = len(val_loader.dataset) if hasattr(val_loader, "dataset") else -1
    print(f"[data] Train: {train_count:,}   Val: {val_count:,}")
    print(f"[augs] Registered {len(augmentations)} candidates:")
    for idx, aug in enumerate(augmentations, start=1):
        tag = " [XAI-driven]" if "icd" in aug.name else ""
        print(f"      {idx:2d}. {aug.name:<24s} (p={aug.probability:.2f}){tag}")
    print()

    reporter = Reporter(config.report_dir)
    trainer = BNNRTrainer(
        model=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=augmentations,
        config=config,
        reporter=reporter,
    )
    result = trainer.run()

    events_path = result.report_json_path.parent / "events.jsonl"
    print()
    print("=" * 68)
    print("  YOLO COCO128 Detection Showcase — Results")
    print("-" * 68)
    print(f"  Best path      : {result.best_path}")
    print(f"  Best metrics   : {result.best_metrics}")
    print(f"  Report JSON    : {result.report_json_path}")
    print(f"  Events (JSONL) : {events_path}")
    if args.with_dashboard:
        print(f"  Dashboard      : {dashboard_url}")
    print("=" * 68)
    print()

    if args.with_dashboard:
        print("Dashboard is still running — press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
