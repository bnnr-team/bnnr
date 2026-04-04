"""Tests for bnnr.data_quality — duplicate detection + image sanity checks."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bnnr.data_quality import (
    check_image_quality_batch,
    compute_dhash_batch,
    find_near_duplicates,
    run_data_quality_analysis,
)

# ============================================================
#  compute_dhash_batch
# ============================================================

class TestComputeDhashBatch:
    """dHash should be deterministic and distinguish different images."""

    def test_deterministic(self) -> None:
        imgs = torch.rand(4, 3, 32, 32)
        h1 = compute_dhash_batch(imgs)
        h2 = compute_dhash_batch(imgs)
        np.testing.assert_array_equal(h1, h2)

    def test_identical_images_same_hash(self) -> None:
        img = torch.rand(1, 3, 64, 64)
        batch = img.repeat(3, 1, 1, 1)
        hashes = compute_dhash_batch(batch)
        assert hashes[0] == hashes[1] == hashes[2]

    def test_different_images_different_hash(self) -> None:
        a = torch.zeros(1, 3, 32, 32)
        b = torch.ones(1, 3, 32, 32)
        c = torch.rand(1, 3, 32, 32)
        hashes = compute_dhash_batch(torch.cat([a, b, c]))
        # At least two of three should differ
        unique = len(set(hashes.tolist()))
        assert unique >= 2

    def test_output_dtype(self) -> None:
        imgs = torch.rand(2, 3, 16, 16)
        hashes = compute_dhash_batch(imgs)
        assert hashes.dtype == np.uint64

    def test_grayscale_input(self) -> None:
        imgs = torch.rand(3, 1, 32, 32)
        hashes = compute_dhash_batch(imgs)
        assert hashes.shape == (3,)

    def test_small_images(self) -> None:
        """Should work even for very small images (8×8)."""
        imgs = torch.rand(2, 3, 8, 8)
        hashes = compute_dhash_batch(imgs)
        assert hashes.shape == (2,)


# ============================================================
#  find_near_duplicates
# ============================================================

class TestFindNearDuplicates:
    """Test duplicate-pair discovery from hash arrays."""

    def test_exact_duplicates(self) -> None:
        hashes = np.array([100, 200, 100, 300, 200], dtype=np.uint64)
        indices = np.arange(5)
        pairs = find_near_duplicates(hashes, indices, threshold=0)
        idx_pairs = {(a, b) for a, b, _ in pairs}
        assert (0, 2) in idx_pairs
        assert (1, 4) in idx_pairs

    def test_no_duplicates(self) -> None:
        # Maximally different hashes
        hashes = np.array([0, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA], dtype=np.uint64)
        indices = np.arange(3)
        pairs = find_near_duplicates(hashes, indices, threshold=5)
        assert len(pairs) == 0

    def test_near_duplicates_within_threshold(self) -> None:
        base = np.uint64(0)
        near = np.uint64(0b111)  # hamming distance = 3
        hashes = np.array([base, near], dtype=np.uint64)
        indices = np.arange(2)
        pairs = find_near_duplicates(hashes, indices, threshold=5)
        assert len(pairs) == 1
        assert pairs[0][2] == 3

    def test_empty_input(self) -> None:
        hashes = np.array([], dtype=np.uint64)
        indices = np.array([], dtype=np.int64)
        pairs = find_near_duplicates(hashes, indices)
        assert pairs == []


# ============================================================
#  check_image_quality_batch
# ============================================================

class TestCheckImageQualityBatch:
    """Test per-image quality checks."""

    def test_normal_images_no_issues(self) -> None:
        imgs = torch.rand(4, 3, 32, 32) * 0.5 + 0.25  # mean ≈ 0.5, good std
        issues = check_image_quality_batch(imgs, batch_offset=0)
        assert len(issues) == 0

    def test_zero_variance_detected(self) -> None:
        solid = torch.full((1, 3, 32, 32), 0.5)
        issues = check_image_quality_batch(solid, batch_offset=0)
        types = [iss for entry in issues for iss in entry["issues"]]
        assert "zero_variance" in types

    def test_nan_detected(self) -> None:
        img = torch.rand(1, 3, 32, 32)
        img[0, 0, 0, 0] = float("nan")
        issues = check_image_quality_batch(img, batch_offset=0)
        types = [iss for entry in issues for iss in entry["issues"]]
        assert "contains_nan" in types

    def test_inf_detected(self) -> None:
        img = torch.rand(1, 3, 32, 32)
        img[0, 1, 5, 5] = float("inf")
        issues = check_image_quality_batch(img, batch_offset=0)
        types = [iss for entry in issues for iss in entry["issues"]]
        assert "contains_inf" in types

    def test_near_black_detected(self) -> None:
        img = torch.full((1, 3, 32, 32), 0.001)
        issues = check_image_quality_batch(img, batch_offset=0)
        types = [iss for entry in issues for iss in entry["issues"]]
        assert "near_black" in types

    def test_near_white_detected(self) -> None:
        img = torch.full((1, 3, 32, 32), 0.999)
        issues = check_image_quality_batch(img, batch_offset=0)
        types = [iss for entry in issues for iss in entry["issues"]]
        assert "near_white" in types

    def test_batch_offset_applied(self) -> None:
        solid = torch.full((1, 3, 32, 32), 0.5)
        issues = check_image_quality_batch(solid, batch_offset=100)
        assert issues[0]["index"] == 100

    def test_mixed_batch(self) -> None:
        good = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
        bad = torch.full((1, 3, 32, 32), 0.5)
        batch = torch.cat([good, bad])
        issues = check_image_quality_batch(batch, batch_offset=0)
        # Only the solid image (index 2) should be flagged
        flagged_indices = [e["index"] for e in issues]
        assert 2 in flagged_indices
        assert 0 not in flagged_indices
        assert 1 not in flagged_indices


# ============================================================
#  run_data_quality_analysis (integration)
# ============================================================

class TestRunDataQualityAnalysis:
    """End-to-end test with a DataLoader."""

    def _make_loader(self, images: torch.Tensor, labels: torch.Tensor) -> DataLoader:
        return DataLoader(TensorDataset(images, labels), batch_size=4)

    def test_clean_dataset(self) -> None:
        imgs = torch.rand(20, 3, 32, 32)
        labels = torch.randint(0, 3, (20,))
        loader = self._make_loader(imgs, labels)
        result = run_data_quality_analysis(loader)
        dq = result["data_quality"]
        assert dq["scanned_samples"] == 20
        assert isinstance(dq["warnings"], list)
        assert isinstance(dq["summary"], str)

    def test_duplicate_detection(self) -> None:
        base = torch.rand(1, 3, 32, 32)
        # 5 exact copies + 5 random
        imgs = torch.cat([base.repeat(5, 1, 1, 1), torch.rand(5, 3, 32, 32)])
        labels = torch.randint(0, 2, (10,))
        loader = self._make_loader(imgs, labels)
        result = run_data_quality_analysis(loader)
        dq = result["data_quality"]
        # Should find at least one duplicate group
        assert dq["total_duplicate_pairs"] > 0
        dup_warning = [w for w in dq["warnings"] if w["type"] == "near_duplicates"]
        assert len(dup_warning) == 1

    def test_quality_issues_detected(self) -> None:
        good = torch.rand(5, 3, 32, 32) * 0.5 + 0.25
        bad = torch.full((2, 3, 32, 32), 0.5)  # zero variance
        imgs = torch.cat([good, bad])
        labels = torch.randint(0, 2, (7,))
        loader = self._make_loader(imgs, labels)
        result = run_data_quality_analysis(loader)
        dq = result["data_quality"]
        assert dq["total_flagged_images"] >= 2
        zero_var = [w for w in dq["warnings"] if w["type"] == "zero_variance"]
        assert len(zero_var) == 1
        assert zero_var[0]["count"] >= 2

    def test_max_samples_cap(self) -> None:
        imgs = torch.rand(50, 3, 16, 16)
        labels = torch.randint(0, 2, (50,))
        loader = self._make_loader(imgs, labels)
        result = run_data_quality_analysis(loader, max_samples=10)
        dq = result["data_quality"]
        assert dq["scanned_samples"] == 10
