"""Tests for example dataset auto-fetch helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bnnr.example_data import ensure_coco128_yolo, resolve_yolo_example_data_yaml


def test_ensure_coco128_skips_when_complete(tmp_path: Path) -> None:
    coco = tmp_path / "coco128"
    img_dir = coco / "images" / "train2017"
    img_dir.mkdir(parents=True)
    (img_dir / "stub.jpg").write_bytes(b"\xff\xd8\xff")
    yaml_path = coco / "data.yaml"
    yaml_path.write_text(
        "train: images/train2017\nval: images/train2017\nnc: 1\n",
        encoding="utf-8",
    )
    out = ensure_coco128_yolo(coco)
    assert out == yaml_path
    assert out.read_text(encoding="utf-8").startswith("train:")


def test_resolve_yolo_returns_existing_file(tmp_path: Path) -> None:
    y = tmp_path / "data.yaml"
    y.write_text("train: images/t\nval: images/t\nnc: 1\n", encoding="utf-8")
    assert resolve_yolo_example_data_yaml(y) == y.resolve()


def test_resolve_yolo_missing_non_coco128_raises(tmp_path: Path) -> None:
    missing = tmp_path / "other" / "data.yaml"
    with pytest.raises(FileNotFoundError, match="YOLO data.yaml not found"):
        resolve_yolo_example_data_yaml(missing, auto_download=True)


def test_resolve_yolo_no_auto_download(tmp_path: Path) -> None:
    missing = tmp_path / "coco128" / "data.yaml"
    with pytest.raises(FileNotFoundError):
        resolve_yolo_example_data_yaml(missing, auto_download=False)
