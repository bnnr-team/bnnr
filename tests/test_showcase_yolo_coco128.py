"""Smoke tests for YOLO COCO128 showcase flow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "detection" / "showcase_yolo_coco128.py"
    spec = importlib.util.spec_from_file_location("showcase_yolo_coco128", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_yolo_class_names_from_list(tmp_path) -> None:
    module = _load_module()
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "train: images/train\nval: images/val\nnames: [person, bicycle, car]\n",
        encoding="utf-8",
    )
    spec = module._load_yolo_data_spec(yaml_path)
    names = module._read_yolo_class_names(spec)
    assert names == ["person", "bicycle", "car"]


def test_read_yolo_class_names_from_dict_and_nc(tmp_path) -> None:
    module = _load_module()
    yaml_dict = tmp_path / "dict.yaml"
    yaml_dict.write_text(
        "train: images/train\nval: images/val\nnames: {0: cat, 1: dog}\n",
        encoding="utf-8",
    )
    spec_dict = module._load_yolo_data_spec(yaml_dict)
    names_dict = module._read_yolo_class_names(spec_dict)
    assert names_dict == ["cat", "dog"]

    yaml_nc = tmp_path / "nc.yaml"
    yaml_nc.write_text(
        "train: images/train\nval: images/val\nnc: 3\n",
        encoding="utf-8",
    )
    spec_nc = module._load_yolo_data_spec(yaml_nc)
    names_nc = module._read_yolo_class_names(spec_nc)
    assert names_nc == ["class_0", "class_1", "class_2"]


def test_load_yolo_data_spec_rejects_bnnr_config_yaml(tmp_path) -> None:
    module = _load_module()
    wrong_yaml = tmp_path / "bnnr_config.yaml"
    wrong_yaml.write_text(
        "task: detection\nm_epochs: 1\nmax_iterations: 1\nselection_metric: map_50\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"looks like a BNNR config YAML"):
        module._load_yolo_data_spec(wrong_yaml)
