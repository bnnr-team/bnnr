
"""BNNR + Ultralytics YOLOv8 quickstart on COCO128 (UltralyticsDetectionAdapter).

This uses BNNR's Python API — not the ``yolo train`` CLI. Images stay in [0, 1] float.

Install:
    pip install "bnnr[ultralytics]"

Run:
    PYTHONPATH=src python examples/integrations/ultralytics_yolo_quickstart.py --quick

Citation: cite BNNR (and Ultralytics per their license) — see docs/citation.md in the bnnr repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as script from repo root
_INTEGRATIONS = Path(__file__).resolve().parent
if str(_INTEGRATIONS) not in sys.path:
    sys.path.insert(0, str(_INTEGRATIONS))

from coco128_ultralytics_dataset import (  # noqa: E402
    Coco128BnnrDataset,
    coco80_names,
    split_coco128_paths,
)

try:
    import ultralytics
    from bnnr import BNNRConfig, BNNRTrainer, start_dashboard
    from bnnr.detection_adapter import UltralyticsDetectionAdapter
    from bnnr.detection_augmentations import DetectionHorizontalFlip, DetectionRandomScale
    from bnnr.detection_collate import detection_collate_fn_with_index
    from bnnr.detection_icd import DetectionAICD, DetectionICD
    from bnnr.example_data import ensure_coco128_yolo
except ImportError as exc:
    print(
        'Missing dependencies. Install with:\n  pip install "bnnr[ultralytics]"',
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

SEED = 42
TARGET_SIZE = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BNNR + Ultralytics YOLOv8 on COCO128 (bbox-aware aug + DetectionICD/AICD)",
    )
    p.add_argument("--quick", action="store_true", help="Short run for smoke tests")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=Path("data/coco128"))
    p.add_argument("--no-auto-download", action="store_true")
    p.add_argument("--with-dashboard", action="store_true")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/ultralytics_yolo_quickstart"),
    )
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics weights path or name")
    return p.parse_args()


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _build_augmentations(quick: bool, seed: int) -> list:
    augs = [
        DetectionHorizontalFlip(probability=0.5, name_override="det_hflip", random_state=seed),
        DetectionICD(
            probability=0.5,
            threshold_percentile=70,
            tile_size=8,
            fill_strategy="gaussian_blur",
            name_override="det_icd",
            random_state=seed + 10,
        ),
        DetectionAICD(
            probability=0.5,
            threshold_percentile=70,
            tile_size=8,
            fill_strategy="gaussian_blur",
            name_override="det_aicd",
            random_state=seed + 11,
        ),
    ]
    if not quick:
        augs.append(
            DetectionRandomScale(
                scale_range=(0.85, 1.15),
                probability=0.5,
                name_override="det_scale",
                random_state=seed + 3,
            ),
        )
    return augs


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print()
    print("=" * 68)
    print("  BNNR + Ultralytics YOLOv8 (not yolo CLI). Images stay in [0, 1] float.")
    print("=" * 68)

    coco_root = args.data_dir.resolve()
    if not args.no_auto_download:
        ensure_coco128_yolo(coco_root)
    elif not (coco_root / "data.yaml").is_file():
        print(
            f"[error] Missing {coco_root / 'data.yaml'} (drop --no-auto-download to fetch COCO128)"
        )
        raise SystemExit(2)

    train_paths, val_paths = split_coco128_paths(coco_root)
    if args.quick:
        train_paths = train_paths[:32]
        val_paths = val_paths[:16]

    train_ds = Coco128BnnrDataset(train_paths, coco_root, target_size=TARGET_SIZE)
    val_ds = Coco128BnnrDataset(val_paths, coco_root, target_size=TARGET_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn_with_index,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn_with_index,
        num_workers=0,
    )

    class_names = coco80_names()
    adapter = UltralyticsDetectionAdapter(
        model_name=args.model,
        device=device,
        num_classes=80,
        lr=1e-3,
    )

    overrides: dict = {
        "task": "detection",
        "device": device,
        "detection_class_names": class_names,
        "report_dir": str(args.report_dir),
        "event_log_enabled": args.with_dashboard,
        "xai_enabled": True,
        "detection_xai_method": "activation",
        "save_checkpoints": False,
        "seed": SEED,
    }
    if args.quick:
        overrides.update(
            {
                "m_epochs": 1,
                "max_iterations": 1,
                "candidate_pruning_enabled": True,
                "candidate_pruning_warmup_epochs": 1,
            }
        )
    else:
        overrides.update(
            {
                "m_epochs": 2,
                "max_iterations": 2,
                "candidate_pruning_enabled": True,
                "candidate_pruning_warmup_epochs": 1,
            }
        )

    config = BNNRConfig(**overrides)
    augmentations = _build_augmentations(args.quick, SEED)

    dashboard_url: str | None = None
    if args.with_dashboard:
        dashboard_url = start_dashboard(config.report_dir)

    print(f"  Device:     {device}")
    print(f"  Model:      {args.model}")
    print(f"  Train/val:  {len(train_ds)} / {len(val_ds)} images")
    print(f"  Report:     {config.report_dir}")
    if dashboard_url:
        print(f"  Dashboard:  {dashboard_url}")
    print()

    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
    result = trainer.run()

    print(f"Done. best_metrics={result.best_metrics}")
    print(f"Report: {result.report_json_path}")


if __name__ == "__main__":
    main()
