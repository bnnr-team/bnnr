"""Auto-fetch small public datasets used by repository examples (YOLO COCO128, etc.)."""

from __future__ import annotations

import logging
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Official Ultralytics mirror (see ultralytics/cfg/datasets/coco128.yaml).
COCO128_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

# The zip contains images/labels only; Ultralytics expects a sibling yaml (we ship a minimal one).
_COCO128_DATA_YAML = """train: images/train2017
val: images/train2017
nc: 80
"""


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "bnnr-examples/1.0 (dataset auto-fetch)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — curated URL
        dest.write_bytes(resp.read())


def ensure_coco128_yolo(coco128_root: Path) -> Path:
    """Ensure Ultralytics COCO128 exists under *coco128_root*; return ``data.yaml`` path.

    *coco128_root* is the dataset directory (e.g. ``data/coco128``), which must contain
    ``images/train2017`` and ``labels/train2017`` after extraction.

    Idempotent: if ``data.yaml`` exists and training images are present, does nothing.
    """
    coco128_root = coco128_root.resolve()
    yaml_path = coco128_root / "data.yaml"
    image_dir = coco128_root / "images" / "train2017"

    if yaml_path.is_file() and image_dir.is_dir() and any(image_dir.glob("*.jpg")):
        return yaml_path

    print(f"[data] Downloading Ultralytics COCO128 (~7 MB) into {coco128_root.parent} …", flush=True)

    parent = coco128_root.parent
    parent.mkdir(parents=True, exist_ok=True)

    zip_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)
        logger.info("Downloading COCO128 from %s", COCO128_ZIP_URL)
        _download_file(COCO128_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(parent)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Failed to download or extract COCO128 from {COCO128_ZIP_URL!r}: {exc}",
        ) from exc
    finally:
        if zip_path is not None:
            zip_path.unlink(missing_ok=True)

    if not image_dir.is_dir() or not any(image_dir.glob("*.jpg")):
        raise RuntimeError(
            f"COCO128 archive did not produce expected layout under {coco128_root} "
            "(missing images/train2017/*.jpg).",
        )

    if not yaml_path.is_file():
        yaml_path.write_text(_COCO128_DATA_YAML, encoding="utf-8")

    return yaml_path


def resolve_yolo_example_data_yaml(requested: Path, *, auto_download: bool = True) -> Path:
    """Return a usable path to YOLO ``data.yaml``, downloading COCO128 when applicable.

    If *requested* exists, returns it unchanged.

    If missing and *requested* looks like ``.../coco128/data.yaml``, attempts
    :func:`ensure_coco128_yolo` on the parent directory when *auto_download* is true.

    Raises
    ------
    FileNotFoundError
        When the path is missing and cannot be fetched automatically.
    """
    requested = requested.resolve()
    if requested.is_file():
        return requested

    is_coco128_layout = requested.name == "data.yaml" and requested.parent.name == "coco128"
    if auto_download and is_coco128_layout:
        return ensure_coco128_yolo(requested.parent)

    if not requested.exists():
        raise FileNotFoundError(
            f"YOLO data.yaml not found: {requested}. "
            "Use a valid path, or default ``data/coco128/data.yaml`` to auto-download COCO128, "
            "or pass --no-auto-download only when data is already on disk.",
        )

    raise FileNotFoundError(f"YOLO data path is not a file: {requested}")
