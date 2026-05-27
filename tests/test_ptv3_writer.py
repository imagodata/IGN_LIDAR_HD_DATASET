"""Unit tests for Ptv3DatasetWriter and split assignment helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ign_lidar.io.formatters import (
    Ptv3DatasetWriter,
    Ptv3Formatter,
    assign_splits_by_tile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _patch(patch_id: str, tile_id: str, n: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "points": rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        "labels": rng.choice([2, 3, 5, 6, 9], size=n).astype(np.int32),
        "intensity": rng.integers(0, 65535, size=n).astype(np.float32),
        "return_number": np.ones(n, dtype=np.float32),
        "normals": np.tile(np.array([0, 0, 1], dtype=np.float32), (n, 1)),
        "patch_id": patch_id,
        "tile_id": tile_id,
    }


# ---------------------------------------------------------------------------
# assign_splits_by_tile
# ---------------------------------------------------------------------------

class TestSplitAssignment:
    def test_deterministic_with_seed(self):
        tiles = [f"t{i:04d}" for i in range(50)]
        a = assign_splits_by_tile(tiles, train=0.8, val=0.1, test=0.1, seed=42)
        b = assign_splits_by_tile(tiles, train=0.8, val=0.1, test=0.1, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        tiles = [f"t{i:04d}" for i in range(50)]
        a = assign_splits_by_tile(tiles, seed=42)
        b = assign_splits_by_tile(tiles, seed=999)
        assert a != b

    def test_ratios_respected(self):
        tiles = [f"t{i:04d}" for i in range(100)]
        a = assign_splits_by_tile(tiles, train=0.8, val=0.1, test=0.1, seed=42)
        counts = {s: sum(1 for v in a.values() if v == s) for s in ("train", "val", "test")}
        assert counts == {"train": 80, "val": 10, "test": 10}

    def test_duplicates_collapsed(self):
        tiles = ["t1", "t1", "t2", "t2", "t3"]
        a = assign_splits_by_tile(tiles, seed=0)
        assert set(a.keys()) == {"t1", "t2", "t3"}

    def test_empty_input(self):
        assert assign_splits_by_tile([], seed=0) == {}

    def test_invalid_ratios_rejected(self):
        with pytest.raises(ValueError, match="sum to 1"):
            assign_splits_by_tile(["t1"], train=0.5, val=0.3, test=0.3)
        with pytest.raises(ValueError, match="non-negative"):
            assign_splits_by_tile(["t1"], train=1.2, val=-0.1, test=-0.1)

    def test_small_n_no_test(self):
        # Only 3 tiles, default ratios → val and test may round to 0
        a = assign_splits_by_tile(["t1", "t2", "t3"], train=0.8, val=0.1, test=0.1, seed=0)
        assert set(a.values()).issubset({"train", "val", "test"})
        assert len(a) == 3


# ---------------------------------------------------------------------------
# Ptv3DatasetWriter
# ---------------------------------------------------------------------------

class TestDatasetWriter:
    def test_writes_to_correct_split(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        splits = {"tile_A": "train", "tile_B": "val", "tile_C": "test"}
        writer = Ptv3DatasetWriter(formatter, tmp_path, splits)

        writer.write_patch(_patch("p1", "tile_A"))
        writer.write_patch(_patch("p2", "tile_B"))
        writer.write_patch(_patch("p3", "tile_C"))

        assert (tmp_path / "train" / "p1").is_dir()
        assert (tmp_path / "val" / "p2").is_dir()
        assert (tmp_path / "test" / "p3").is_dir()
        assert writer.counts == {"train": 1, "val": 1, "test": 1}

    def test_no_tile_leakage_across_splits(self, tmp_path: Path):
        """Patches from the same tile must land in a single split."""
        formatter = Ptv3Formatter()
        # 10 patches per tile, 5 tiles total
        tile_ids = [f"tile_{i}" for i in range(5)]
        splits = assign_splits_by_tile(tile_ids, train=0.6, val=0.2, test=0.2, seed=7)
        writer = Ptv3DatasetWriter(formatter, tmp_path, splits)

        for tile in tile_ids:
            for p in range(10):
                writer.write_patch(_patch(f"{tile}_p{p}", tile))
        writer.finalize()

        # For each tile, collect which split dir holds its patches
        for tile in tile_ids:
            found_splits = set()
            for split_name in ("train", "val", "test"):
                if any((tmp_path / split_name).glob(f"{tile}_*")):
                    found_splits.add(split_name)
            assert len(found_splits) == 1, f"tile {tile} leaked across {found_splits}"

    def test_finalize_writes_meta(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        splits = {"tA": "train", "tB": "val"}
        writer = Ptv3DatasetWriter(formatter, tmp_path, splits)
        writer.write_patch(_patch("p1", "tA"))
        writer.write_patch(_patch("p2", "tA"))
        writer.write_patch(_patch("p3", "tB"))
        counts = writer.finalize()

        assert counts == {"train": 2, "val": 1, "test": 0}
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["splits"] == counts

    def test_unknown_tile_raises_by_default(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"})
        with pytest.raises(KeyError, match="not found in split_assignment"):
            writer.write_patch(_patch("p1", "tB"))

    def test_unknown_tile_skip_mode(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"}, on_unknown="skip")
        target = writer.write_patch(_patch("p1", "tB"))
        assert target is None
        assert writer.counts == {"train": 0, "val": 0, "test": 0}

    def test_unknown_tile_test_mode(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"}, on_unknown="test")
        writer.write_patch(_patch("p1", "tB"))
        assert (tmp_path / "test" / "p1").is_dir()
        assert writer.counts["test"] == 1

    def test_finalize_idempotent(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"})
        writer.write_patch(_patch("p1", "tA"))
        c1 = writer.finalize()
        c2 = writer.finalize()
        assert c1 == c2

    def test_write_after_finalize_raises(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"})
        writer.finalize()
        with pytest.raises(RuntimeError, match="finalize"):
            writer.write_patch(_patch("p1", "tA"))

    def test_invalid_split_value_rejected(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        with pytest.raises(ValueError, match="invalid splits"):
            Ptv3DatasetWriter(formatter, tmp_path, {"tA": "validation"})

    def test_missing_split_key_raises(self, tmp_path: Path):
        formatter = Ptv3Formatter()
        writer = Ptv3DatasetWriter(formatter, tmp_path, {"tA": "train"})
        bad = _patch("p1", "tA")
        del bad["tile_id"]
        with pytest.raises(KeyError, match="split key"):
            writer.write_patch(bad)

    def test_patch_level_split(self, tmp_path: Path):
        """Smoke test: use patch_id as the split key (no tile awareness)."""
        formatter = Ptv3Formatter()
        splits = {"p1": "train", "p2": "val"}
        writer = Ptv3DatasetWriter(formatter, tmp_path, splits, split_key="patch_id")
        writer.write_patch(_patch("p1", "tA"))
        writer.write_patch(_patch("p2", "tA"))
        assert (tmp_path / "train" / "p1").is_dir()
        assert (tmp_path / "val" / "p2").is_dir()
