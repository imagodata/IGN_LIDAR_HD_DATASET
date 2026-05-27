"""Integration test: OutputWriter routes "ptv3_pointcept" through Ptv3DatasetWriter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from ign_lidar.core.output_writer import OutputWriter, PTV3_FORMAT


def _make_config(output_format: str = PTV3_FORMAT) -> OmegaConf:
    """Minimal config compatible with OutputWriter's reads."""
    return OmegaConf.create(
        {
            "processor": {
                "output_format": output_format,
                "processing_mode": "patches_only",
                "architecture": "ptv3",
                "lod_level": "LOD2",
                "patch_size": 150.0,
            },
            "output": {
                "ptv3": {
                    "feat_keys": ["intensity", "return_number", "normals"],
                    "label_schema": "lidar_hd_7cl_contiguous",
                    "layout": "folder",
                    "center_xy": True,
                    "anchor_z_min": True,
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
                },
            },
        }
    )


def _make_patch(n: int = 64, idx: int = 0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "points": rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        "labels": rng.choice([2, 3, 5, 6, 9], size=n).astype(np.int32),
        "intensity": rng.integers(0, 65535, size=n).astype(np.float32),
        "return_number": np.ones(n, dtype=np.float32),
        "normals": np.tile(np.array([0, 0, 1], dtype=np.float32), (n, 1)),
        "_patch_idx": idx,
        "_version": "original",
    }


class TestOutputWriterPtv3Integration:
    def test_save_routes_to_ptv3_layout(self, tmp_path: Path):
        writer = OutputWriter(_make_config())
        patches = [_make_patch(idx=i, seed=i) for i in range(3)]
        laz_file = tmp_path / "LHD_FXX_0942_6543_2024.laz"
        # No real LAZ needed — OutputWriter only uses laz_file.stem
        laz_file.touch()

        writer.save_patches(patches, laz_file, tmp_path / "out")
        summary = writer.finalize()

        # Layout assertions
        ds_root = tmp_path / "out" / "ptv3_dataset"
        assert (ds_root / "meta.json").exists()
        total_dirs = sum(
            1
            for split in ("train", "val", "test")
            for _ in (ds_root / split).glob("*/")
            if (ds_root / split).exists()
        )
        assert total_dirs == 3, f"expected 3 patch dirs total, found {total_dirs}"
        assert sum(summary["ptv3"].values()) == 3

    def test_finalize_writes_meta_with_counts(self, tmp_path: Path):
        writer = OutputWriter(_make_config())
        patches = [_make_patch(idx=i, seed=i) for i in range(5)]
        laz_file = tmp_path / "LHD_FXX_0001_0001_2024.laz"
        laz_file.touch()

        writer.save_patches(patches, laz_file, tmp_path / "out")
        writer.finalize()

        meta = json.loads((tmp_path / "out" / "ptv3_dataset" / "meta.json").read_text())
        assert meta["formatter"] == "Ptv3Formatter"
        assert meta["feat_keys"] == ["intensity", "return_number", "normals"]
        # All 5 patches share the same tile_id → all in the same split (hash determines which)
        non_zero_splits = [s for s, n in meta["splits"].items() if n > 0]
        assert len(non_zero_splits) == 1
        assert sum(meta["splits"].values()) == 5

    def test_finalize_idempotent(self, tmp_path: Path):
        writer = OutputWriter(_make_config())
        patches = [_make_patch(idx=0, seed=0)]
        laz_file = tmp_path / "LHD_FXX_0001_0001_2024.laz"
        laz_file.touch()
        writer.save_patches(patches, laz_file, tmp_path / "out")
        a = writer.finalize()
        b = writer.finalize()
        assert a == b

    def test_no_ptv3_when_format_absent(self, tmp_path: Path):
        """OutputWriter must not create a ptv3_dataset when format=='npz'."""
        writer = OutputWriter(_make_config(output_format="npz"))
        # Don't bother actually saving — we only care that no PTv3 writer is set up.
        assert writer._ptv3_writer is None
        assert writer._ptv3_requested is False
        # finalize is safe to call even with no PTv3 active
        assert writer.finalize() == {}

    def test_multiple_tiles_spread_across_splits(self, tmp_path: Path):
        """With enough distinct tile_ids the hash assigner reaches all 3 splits."""
        writer = OutputWriter(_make_config())
        out = tmp_path / "out"

        # 30 distinct tiles, 1 patch each — large enough to hit train/val/test
        for tile_idx in range(30):
            laz_file = tmp_path / f"LHD_FXX_{tile_idx:04d}_6543_2024.laz"
            laz_file.touch()
            writer.save_patches([_make_patch(idx=0, seed=tile_idx)], laz_file, out)
        summary = writer.finalize()

        counts = summary["ptv3"]
        assert sum(counts.values()) == 30
        # We expect train to dominate (80%) but val and test should not both be 0
        assert counts["train"] > 0
        assert counts["val"] + counts["test"] > 0
