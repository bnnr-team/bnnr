"""Tests for detection-aware augmentations (PR2).

Covers:
- BboxAwareAugmentation interface
- DetectionHorizontalFlip / VerticalFlip / Rotate90 / Scale
- AlbumentationsBboxAugmentation
- MosaicAugmentation / DetectionMixUp
- DetectionICD / DetectionAICD
- get_detection_preset
- BNNRTrainer integration with detection augmentations
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from bnnr.detection_augmentations import (
    AlbumentationsBboxAugmentation,
    BboxAwareAugmentation,
    DetectionHorizontalFlip,
    DetectionMixUp,
    DetectionRandomRotate90,
    DetectionRandomScale,
    DetectionVerticalFlip,
    MosaicAugmentation,
    get_detection_preset,
)
from bnnr.detection_icd import DetectionAICD, DetectionICD

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> np.ndarray:
    """64×64 RGB uint8 image."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_target() -> dict[str, Tensor]:
    """Target dict with 2 boxes and labels."""
    return {
        "boxes": torch.tensor([
            [10.0, 10.0, 30.0, 30.0],
            [40.0, 40.0, 60.0, 60.0],
        ]),
        "labels": torch.tensor([1, 2]),
    }


@pytest.fixture
def sample_target_np() -> dict[str, np.ndarray]:
    """Target dict with numpy arrays."""
    return {
        "boxes": np.array([
            [10.0, 10.0, 30.0, 30.0],
            [40.0, 40.0, 60.0, 60.0],
        ], dtype=np.float32),
        "labels": np.array([1, 2], dtype=np.int64),
    }


def _make_pool(n: int = 10, size: int = 64) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Create a pool of random images and targets for mosaic/mixup."""
    images = [np.random.randint(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]
    targets = [
        {
            "boxes": np.array([[5.0, 5.0, 25.0, 25.0]], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        for _ in range(n)
    ]
    return images, targets


# ---------------------------------------------------------------------------
#  DetectionHorizontalFlip
# ---------------------------------------------------------------------------


class TestDetectionHorizontalFlip:
    def test_flip_changes_image(self, sample_image, sample_target) -> None:
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == sample_image.shape
        assert not np.array_equal(out_img, sample_image)

    def test_flip_transforms_boxes(self, sample_image, sample_target) -> None:
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        # Original box [10, 10, 30, 30] on 64-wide image → [34, 10, 54, 30]
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        assert boxes.shape == (2, 4)
        # x coords should be reflected around width center
        assert boxes[0, 0] == pytest.approx(34.0, abs=1)
        assert boxes[0, 2] == pytest.approx(54.0, abs=1)

    def test_flip_preserves_labels(self, sample_image, sample_target) -> None:
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        labels = out_tgt["labels"]
        if isinstance(labels, Tensor):
            labels = labels.numpy()
        np.testing.assert_array_equal(labels, [1, 2])

    def test_flip_probability_zero(self, sample_image, sample_target) -> None:
        aug = DetectionHorizontalFlip(probability=0.0, random_state=42)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        np.testing.assert_array_equal(out_img, sample_image)

    def test_flip_tensor_target_preserved(self, sample_image, sample_target) -> None:
        """Target with Tensor boxes should return Tensor boxes."""
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert isinstance(out_tgt["boxes"], Tensor)
        assert isinstance(out_tgt["labels"], Tensor)

    def test_flip_numpy_target_preserved(self, sample_image, sample_target_np) -> None:
        """Target with np.ndarray boxes should return np.ndarray boxes."""
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target_np)
        assert isinstance(out_tgt["boxes"], np.ndarray)
        assert isinstance(out_tgt["labels"], np.ndarray)

    def test_flip_empty_boxes(self, sample_image) -> None:
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        target = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        out_img, out_tgt = aug.apply_with_targets(sample_image, target)
        assert out_img.shape == sample_image.shape
        assert out_tgt["boxes"].shape == (0, 4)


# ---------------------------------------------------------------------------
#  DetectionVerticalFlip
# ---------------------------------------------------------------------------


class TestDetectionVerticalFlip:
    def test_vflip_transforms_boxes(self, sample_image, sample_target) -> None:
        aug = DetectionVerticalFlip(probability=1.0, random_state=42)
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        # Original box [10, 10, 30, 30] on 64-tall image → [10, 34, 30, 54]
        assert boxes[0, 1] == pytest.approx(34.0, abs=1)
        assert boxes[0, 3] == pytest.approx(54.0, abs=1)


# ---------------------------------------------------------------------------
#  DetectionRandomRotate90
# ---------------------------------------------------------------------------


class TestDetectionRandomRotate90:
    def test_rotate_produces_valid_output(self, sample_image, sample_target) -> None:
        aug = DetectionRandomRotate90(probability=1.0, random_state=42)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.ndim == 3
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        # Boxes should still be valid (x2 > x1, y2 > y1 after clipping)
        if len(boxes) > 0:
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (boxes[:, 3] >= boxes[:, 1]).all()


# ---------------------------------------------------------------------------
#  DetectionRandomScale
# ---------------------------------------------------------------------------


class TestDetectionRandomScale:
    def test_scale_changes_image_size(self, sample_image, sample_target) -> None:
        aug = DetectionRandomScale(
            scale_range=(0.5, 0.5), probability=1.0, random_state=42,
        )
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        # Scale should change the image dimensions
        assert out_img.shape[0] == 32  # 64 * 0.5
        assert out_img.shape[1] == 32

    def test_scale_adjusts_boxes(self, sample_image, sample_target) -> None:
        aug = DetectionRandomScale(
            scale_range=(2.0, 2.0), probability=1.0, random_state=42,
        )
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        # Box [10, 10, 30, 30] at 2x scale → [20, 20, 60, 60]
        assert boxes[0, 0] == pytest.approx(20.0, abs=1)
        assert boxes[0, 2] == pytest.approx(60.0, abs=1)


# ---------------------------------------------------------------------------
#  AlbumentationsBboxAugmentation
# ---------------------------------------------------------------------------


class TestAlbumentationsBboxAugmentation:
    def test_albu_bbox_basic(self, sample_image, sample_target) -> None:
        albu = pytest.importorskip("albumentations")

        transform = albu.Compose(
            [albu.HorizontalFlip(p=1.0)],
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.1,
            ),
        )
        aug = AlbumentationsBboxAugmentation(
            transform, probability=1.0, random_state=42,
        )
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == sample_image.shape
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        assert len(boxes) == 2

    def test_albu_bbox_preserves_labels(self, sample_image, sample_target) -> None:
        albu = pytest.importorskip("albumentations")

        transform = albu.Compose(
            [albu.NoOp()],
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
            ),
        )
        aug = AlbumentationsBboxAugmentation(
            transform, probability=1.0,
        )
        _, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        labels = out_tgt["labels"]
        if isinstance(labels, Tensor):
            labels = labels.numpy()
        np.testing.assert_array_equal(labels, [1, 2])


# ---------------------------------------------------------------------------
#  MosaicAugmentation
# ---------------------------------------------------------------------------


class TestMosaicAugmentation:
    def test_mosaic_without_pool_noop(self, sample_image, sample_target) -> None:
        """Without a pool, mosaic should be a no-op."""
        aug = MosaicAugmentation(probability=1.0, random_state=42)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        np.testing.assert_array_equal(out_img, sample_image)

    def test_mosaic_with_pool(self, sample_image, sample_target) -> None:
        aug = MosaicAugmentation(
            output_size=(64, 64), probability=1.0, random_state=42,
        )
        pool_imgs, pool_tgts = _make_pool(n=5, size=64)
        aug.set_pool(pool_imgs, pool_tgts)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == (64, 64, 3)
        # Should have boxes from all 4 quadrants
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        assert len(boxes) > 0


# ---------------------------------------------------------------------------
#  DetectionMixUp
# ---------------------------------------------------------------------------


class TestDetectionMixUp:
    def test_mixup_without_pool_noop(self, sample_image, sample_target) -> None:
        aug = DetectionMixUp(probability=1.0, random_state=42)
        out_img, _ = aug.apply_with_targets(sample_image, sample_target)
        np.testing.assert_array_equal(out_img, sample_image)

    def test_mixup_with_pool(self, sample_image, sample_target) -> None:
        aug = DetectionMixUp(probability=1.0, random_state=42)
        pool_imgs, pool_tgts = _make_pool(n=5, size=64)
        aug.set_pool(pool_imgs, pool_tgts)
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == sample_image.shape
        # Should combine boxes from both images
        boxes = out_tgt["boxes"]
        if isinstance(boxes, Tensor):
            boxes = boxes.numpy()
        assert len(boxes) >= 2  # at least original 2 boxes


# ---------------------------------------------------------------------------
#  DetectionICD / DetectionAICD
# ---------------------------------------------------------------------------


class TestDetectionICD:
    def test_icd_masks_object_regions(self, sample_image, sample_target) -> None:
        aug = DetectionICD(
            probability=1.0,
            threshold_percentile=50.0,
            tile_size=8,
            fill_strategy="solid",
            fill_value=0,
            random_state=42,
        )
        out_img, out_tgt = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == sample_image.shape
        # Image should be modified
        assert not np.array_equal(out_img, sample_image)
        # Targets should be unchanged
        if isinstance(out_tgt["labels"], Tensor):
            np.testing.assert_array_equal(out_tgt["labels"].numpy(), [1, 2])
        else:
            np.testing.assert_array_equal(out_tgt["labels"], [1, 2])

    def test_icd_empty_boxes(self, sample_image) -> None:
        aug = DetectionICD(probability=1.0, random_state=42)
        target = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        out_img, out_tgt = aug.apply_with_targets(sample_image, target)
        # With no boxes, saliency is all zeros → no masking
        assert out_img.shape == sample_image.shape

    def test_aicd_different_from_icd(self, sample_image, sample_target) -> None:
        """ICD and AICD should produce different results (different masks)."""
        icd = DetectionICD(
            probability=1.0, threshold_percentile=50.0,
            fill_strategy="solid", fill_value=0, random_state=42,
        )
        aicd = DetectionAICD(
            probability=1.0, threshold_percentile=50.0,
            fill_strategy="solid", fill_value=0, random_state=42,
        )
        out_icd, _ = icd.apply_with_targets(sample_image.copy(), sample_target)
        out_aicd, _ = aicd.apply_with_targets(sample_image.copy(), sample_target)
        # They mask different regions, so outputs should differ
        assert not np.array_equal(out_icd, out_aicd)


class TestDetectionAICD:
    def test_aicd_masks_background(self, sample_image, sample_target) -> None:
        # threshold_percentile=30 → AICD uses percentile=70 internally,
        # which gives a threshold > 0 so background tiles (saliency=0) get masked.
        aug = DetectionAICD(
            probability=1.0,
            threshold_percentile=30.0,
            tile_size=8,
            fill_strategy="solid",
            fill_value=0,
            random_state=42,
        )
        out_img, _ = aug.apply_with_targets(sample_image, sample_target)
        assert out_img.shape == sample_image.shape
        assert not np.array_equal(out_img, sample_image)


# ---------------------------------------------------------------------------
#  Presets
# ---------------------------------------------------------------------------


class TestDetectionPresets:
    def test_light_preset(self) -> None:
        augs = get_detection_preset("light")
        assert len(augs) >= 1
        assert all(isinstance(a, BboxAwareAugmentation) for a in augs)

    def test_standard_preset(self) -> None:
        augs = get_detection_preset("standard")
        assert len(augs) >= 4  # hflip + vflip + rotate + scale + maybe color

    def test_aggressive_preset(self) -> None:
        augs = get_detection_preset("aggressive")
        # Should include mosaic and mixup
        names = [a.name for a in augs]
        assert "det_mosaic" in names
        assert "det_mixup" in names

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_detection_preset("nonexistent")


# ---------------------------------------------------------------------------
#  BboxAwareAugmentation fallback apply (image-only)
# ---------------------------------------------------------------------------


class TestBboxAwareImageOnly:
    def test_apply_fallback(self, sample_image) -> None:
        """BaseAugmentation.apply should work for image-only path."""
        aug = DetectionHorizontalFlip(probability=1.0, random_state=42)
        out = aug.apply(sample_image)
        assert out.shape == sample_image.shape
        assert not np.array_equal(out, sample_image)
