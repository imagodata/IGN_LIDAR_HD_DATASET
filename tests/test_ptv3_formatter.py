"""Unit tests for Ptv3Formatter (Pointcept-compatible export)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ign_lidar.classification_schema import (
    LIDARHD_7CL_IGNORE_INDEX,
    LIDARHD_7CL_NAMES,
)
from ign_lidar.io.formatters import Ptv3Formatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_patch(n: int = 256, seed: int = 0, patch_id: str = "p0001") -> dict:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=[650_000, 6_860_000, 0.0],
                      high=[650_150, 6_860_150, 35.0],
                      size=(n, 3)).astype(np.float64)
    # Build labels with a representative mix of ASPRS codes.
    asprs = rng.choice([1, 2, 3, 4, 5, 6, 9, 17], size=n,
                       p=[0.05, 0.5, 0.05, 0.05, 0.15, 0.15, 0.04, 0.01])
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    return {
        "points": pts.astype(np.float32),
        "labels": asprs.astype(np.int32),
        "intensity": rng.integers(0, 65535, size=n).astype(np.float32),
        "return_number": rng.integers(1, 5, size=n).astype(np.float32),
        "num_returns": rng.integers(1, 5, size=n).astype(np.float32),
        "normals": normals,
        "patch_id": patch_id,
        "tile_id": "0942_6543",
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_init(self):
        fmt = Ptv3Formatter()
        assert fmt.label_schema == "lidar_hd_7cl_contiguous"
        assert fmt.layout == "folder"
        assert fmt.ignore_index == LIDARHD_7CL_IGNORE_INDEX

    def test_invalid_schema_rejected(self):
        with pytest.raises(ValueError, match="label_schema"):
            Ptv3Formatter(label_schema="cityscapes")

    def test_invalid_layout_rejected(self):
        with pytest.raises(ValueError, match="layout"):
            Ptv3Formatter(layout="zarr")

    def test_unknown_feat_key_rejected(self):
        with pytest.raises(ValueError, match="Unsupported feat key"):
            Ptv3Formatter(feat_keys=("magic_feature",))


# ---------------------------------------------------------------------------
# format_patch — shapes, dtypes, coord transforms
# ---------------------------------------------------------------------------

class TestFormatPatch:
    def test_shapes_and_dtypes(self):
        patch = _synthetic_patch(n=512)
        fmt = Ptv3Formatter(feat_keys=("intensity", "return_number", "normals"))
        out = fmt.format_patch(patch)

        assert out["coord"].shape == (512, 3)
        assert out["coord"].dtype == np.float32
        # 1 + 1 + 3 = 5 feature channels
        assert out["feat"].shape == (512, 5)
        assert out["feat"].dtype == np.float32
        assert out["segment"].shape == (512,)
        assert out["segment"].dtype == np.int16
        assert out["name"] == "p0001"
        assert out["tile_id"] == "0942_6543"
        assert out["feat_layout"] == [
            "intensity", "return_number", "normal_x", "normal_y", "normal_z",
        ]

    def test_xy_centered_z_anchored(self):
        patch = _synthetic_patch(n=256)
        fmt = Ptv3Formatter()
        out = fmt.format_patch(patch)
        coord = out["coord"]
        # XY centroid near zero, Z minimum at zero
        assert np.allclose(coord[:, :2].mean(axis=0), 0.0, atol=1e-3)
        assert np.isclose(coord[:, 2].min(), 0.0, atol=1e-6)

    def test_disable_recentering(self):
        patch = _synthetic_patch(n=128)
        fmt = Ptv3Formatter(center_xy=False, anchor_z_min=False)
        out = fmt.format_patch(patch)
        assert not np.allclose(out["coord"][:, :2].mean(axis=0), 0.0, atol=1.0)

    def test_missing_optional_feature_zero_filled(self):
        patch = _synthetic_patch(n=64)
        del patch["intensity"]
        fmt = Ptv3Formatter(feat_keys=("intensity", "normals"))
        out = fmt.format_patch(patch)
        # intensity column should be all zeros, normals untouched
        assert np.all(out["feat"][:, 0] == 0.0)
        assert np.linalg.norm(out["feat"][:, 1:4], axis=1).mean() > 0.5

    def test_missing_required_keys_raise(self):
        fmt = Ptv3Formatter()
        with pytest.raises(KeyError, match="points"):
            fmt.format_patch({"labels": np.zeros(10)})
        with pytest.raises(KeyError, match="labels"):
            fmt.format_patch({"points": np.zeros((10, 3))})

    def test_feature_length_mismatch_raises(self):
        patch = _synthetic_patch(n=100)
        patch["intensity"] = patch["intensity"][:50]  # truncated
        fmt = Ptv3Formatter(feat_keys=("intensity",))
        with pytest.raises(ValueError, match="length"):
            fmt.format_patch(patch)


# ---------------------------------------------------------------------------
# Label remap — contiguous + ignore_index
# ---------------------------------------------------------------------------

class TestLabelRemap:
    def test_labels_are_contiguous_or_ignored(self):
        patch = _synthetic_patch(n=2048)
        fmt = Ptv3Formatter()
        out = fmt.format_patch(patch)
        unique = set(out["segment"].tolist())
        # Allowed: 0..6 plus -1 (ignore for ASPRS code 1)
        assert unique.issubset(set(range(7)) | {LIDARHD_7CL_IGNORE_INDEX}), unique

    def test_known_codes_map_correctly(self):
        patch = {
            "points": np.zeros((6, 3), dtype=np.float32),
            "labels": np.array([2, 3, 4, 5, 6, 17], dtype=np.int32),
            "patch_id": "tiny",
        }
        fmt = Ptv3Formatter(feat_keys=())
        out = fmt.format_patch(patch)
        # Expected: ground=0, low_veg=1, med_veg=2, high_veg=3, building=4, bridge=6
        assert out["segment"].tolist() == [0, 1, 2, 3, 4, 6]

    def test_raw_schema_preserves_codes(self):
        patch = {
            "points": np.zeros((5, 3), dtype=np.float32),
            "labels": np.array([1, 6, 9, 17, 64], dtype=np.int32),
            "patch_id": "raw_test",
        }
        fmt = Ptv3Formatter(label_schema="raw", feat_keys=())
        out = fmt.format_patch(patch)
        # Raw mode should keep ASPRS codes verbatim — including non-7cl codes
        assert out["segment"].tolist() == [1, 6, 9, 17, 64]


# ---------------------------------------------------------------------------
# Disk I/O — folder + pth layouts
# ---------------------------------------------------------------------------

class TestWritePatch:
    def test_folder_layout(self, tmp_path: Path):
        patch = _synthetic_patch(n=128, patch_id="px001")
        fmt = Ptv3Formatter()
        out = fmt.format_patch(patch)
        tile_dir = fmt.write_patch(out, tmp_path / "train")

        assert tile_dir.is_dir()
        coord = np.load(tile_dir / "coord.npy")
        feat = np.load(tile_dir / "feat.npy")
        segment = np.load(tile_dir / "segment.npy")
        assert coord.shape == (128, 3) and coord.dtype == np.float32
        assert feat.shape[0] == 128 and feat.dtype == np.float32
        assert segment.shape == (128,) and segment.dtype == np.int16

    def test_pth_layout(self, tmp_path: Path):
        torch = pytest.importorskip("torch")
        patch = _synthetic_patch(n=64, patch_id="px002")
        fmt = Ptv3Formatter(layout="pth")
        out = fmt.format_patch(patch)
        target = fmt.write_patch(out, tmp_path / "train")
        assert target.suffix == ".pth"
        blob = torch.load(target, weights_only=False)
        assert blob["coord"].shape == (64, 3)
        assert blob["segment"].dtype == torch.int64


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

class TestMeta:
    def test_meta_records_schema_and_layout(self, tmp_path: Path):
        fmt = Ptv3Formatter(feat_keys=("intensity", "return_number", "normals"))
        target = fmt.write_meta(tmp_path, num_patches_per_split={"train": 80, "val": 10, "test": 10})
        meta = json.loads(target.read_text())

        assert meta["formatter"] == "Ptv3Formatter"
        assert meta["label_names"] == LIDARHD_7CL_NAMES
        assert meta["ignore_index"] == LIDARHD_7CL_IGNORE_INDEX
        assert meta["feat_keys"] == ["intensity", "return_number", "normals"]
        assert meta["feat_dim"] == 5
        assert meta["splits"] == {"train": 80, "val": 10, "test": 10}
        # Round-trip: each contiguous id maps back to a representative ASPRS code
        assert meta["label_to_asprs"]["4"] == 6   # building → ASPRS 6
        assert meta["label_to_asprs"]["6"] == 17  # bridge → ASPRS 17


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

class TestFeatureScaling:
    def test_intensity_normalized_to_unit(self):
        patch = _synthetic_patch(n=256)
        fmt = Ptv3Formatter(feat_keys=("intensity",))
        out = fmt.format_patch(patch)
        assert out["feat"].min() >= 0.0
        assert out["feat"].max() <= 1.0

    def test_return_number_normalized(self):
        patch = _synthetic_patch(n=256)
        fmt = Ptv3Formatter(feat_keys=("return_number",))
        out = fmt.format_patch(patch)
        assert out["feat"].max() <= 1.0
