"""End-to-end smoke test: write a dataset, load it back via a minimal Pointcept-style consumer.

This validates the *contract* with Pointcept (folder layout + meta.json) without
depending on the Pointcept package itself — the consumer here mirrors the
shape and dtype expectations of ``pointcept.datasets.DefaultDataset``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from ign_lidar.classification_schema import LIDARHD_7CL_IGNORE_INDEX
from ign_lidar.io.formatters import (
    HashSplitAssigner,
    Ptv3DatasetWriter,
    Ptv3Formatter,
    assign_splits_by_tile,
)


# ---------------------------------------------------------------------------
# Minimal Pointcept-style consumer
# ---------------------------------------------------------------------------

class MinimalPointceptDataset:
    """Reproduces the read-side contract of ``pointcept.datasets.DefaultDataset``.

    Real PTv3 training loops call ``get_data(idx) -> dict[str, np.ndarray]``
    with the same keys we synthesise here. If this consumer can load every
    sample and pass the invariants below, the layout is Pointcept-ready.
    """

    REQUIRED_KEYS = ("coord", "feat", "segment")

    def __init__(self, data_root: Path, split: str):
        self.data_root = Path(data_root)
        self.split = split
        meta = json.loads((self.data_root / "meta.json").read_text())
        self.num_classes = len(meta["label_names"])
        self.ignore_index = meta["ignore_index"]
        self.feat_dim = meta["feat_dim"]
        split_dir = self.data_root / split
        self.samples: List[Path] = sorted(p for p in split_dir.iterdir() if p.is_dir())

    def __len__(self) -> int:
        return len(self.samples)

    def get_data(self, idx: int) -> Dict[str, np.ndarray]:
        sample_dir = self.samples[idx]
        data = {key: np.load(sample_dir / f"{key}.npy") for key in self.REQUIRED_KEYS}
        data["name"] = sample_dir.name
        return data

    def validate(self) -> None:
        """Raise if any sample violates Pointcept's expectations."""
        for idx in range(len(self)):
            d = self.get_data(idx)
            assert d["coord"].ndim == 2 and d["coord"].shape[1] == 3, d["coord"].shape
            assert d["coord"].dtype == np.float32
            assert d["feat"].shape[0] == d["coord"].shape[0]
            assert d["feat"].shape[1] == self.feat_dim, (d["feat"].shape, self.feat_dim)
            assert d["feat"].dtype == np.float32
            assert d["segment"].shape == (d["coord"].shape[0],)
            assert d["segment"].dtype in (np.int16, np.int32, np.int64)
            valid_mask = d["segment"] != self.ignore_index
            if valid_mask.any():
                lo, hi = d["segment"][valid_mask].min(), d["segment"][valid_mask].max()
                assert 0 <= lo and hi < self.num_classes, (lo, hi, self.num_classes)
            # No NaN/Inf — would crash autograd
            assert np.isfinite(d["coord"]).all()
            assert np.isfinite(d["feat"]).all()


# ---------------------------------------------------------------------------
# Fixture: build a dataset on disk
# ---------------------------------------------------------------------------

def _build_patch(tile_id: str, patch_idx: int, n: int = 256, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # Use realistic Lambert93 coords to exercise the float64 centering path.
    base_x, base_y = 650_000 + 150 * patch_idx, 6_860_000 + 150 * patch_idx
    pts = np.column_stack([
        rng.uniform(base_x, base_x + 150, size=n),
        rng.uniform(base_y, base_y + 150, size=n),
        rng.uniform(0, 40, size=n),
    ]).astype(np.float32)
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    return {
        "points": pts,
        "labels": rng.choice([2, 3, 5, 6, 9, 17], size=n).astype(np.int32),
        "intensity": rng.integers(0, 65535, size=n).astype(np.float32),
        "return_number": rng.integers(1, 5, size=n).astype(np.float32),
        "normals": normals,
        "patch_id": f"{tile_id}_p{patch_idx:04d}",
        "tile_id": tile_id,
    }


@pytest.fixture
def built_dataset(tmp_path: Path) -> Path:
    """Build a small dataset on disk and return its root path."""
    # 8 tiles × 3 patches = 24 patches; ratios 0.5/0.25/0.25 → 4 train / 2 val / 2 test
    tile_ids = [f"tile_{i:03d}" for i in range(8)]
    splits = assign_splits_by_tile(tile_ids, train=0.5, val=0.25, test=0.25, seed=0)
    formatter = Ptv3Formatter(
        feat_keys=("intensity", "return_number", "normals"),
        label_schema="lidar_hd_7cl_contiguous",
    )
    writer = Ptv3DatasetWriter(formatter, tmp_path / "ds", splits)
    for ti, tile in enumerate(tile_ids):
        for p in range(3):  # 3 patches per tile
            writer.write_patch(_build_patch(tile, patch_idx=p, n=256, seed=ti * 10 + p))
    writer.finalize()
    return tmp_path / "ds"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_consumer_can_load_every_sample(built_dataset: Path):
    """A Pointcept-style consumer can iterate every sample without errors."""
    total = 0
    for split in ("train", "val", "test"):
        ds = MinimalPointceptDataset(built_dataset, split=split)
        ds.validate()
        total += len(ds)
    assert total == 24  # 8 tiles × 3 patches


def test_no_tile_leakage_observed_by_consumer(built_dataset: Path):
    """Loading every sample, each tile_id appears in exactly one split."""
    seen: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        ds = MinimalPointceptDataset(built_dataset, split=split)
        for idx in range(len(ds)):
            sample = ds.get_data(idx)
            tile = sample["name"].rsplit("_p", 1)[0]  # tile_001_p0002 -> tile_001
            if tile in seen:
                assert seen[tile] == split, (
                    f"tile {tile} appears in both {seen[tile]} and {split}"
                )
            seen[tile] = split


def test_label_distribution_is_sane(built_dataset: Path):
    """Across all splits, labels are contiguous and ignore_index is honoured."""
    all_labels = []
    for split in ("train", "val", "test"):
        ds = MinimalPointceptDataset(built_dataset, split=split)
        for idx in range(len(ds)):
            all_labels.append(ds.get_data(idx)["segment"])
    cat = np.concatenate(all_labels)
    unique = set(cat.tolist())
    # Every observed code must be either contiguous in [0..6] or the sentinel
    assert unique.issubset(set(range(7)) | {LIDARHD_7CL_IGNORE_INDEX}), unique
    # We seeded enough variety that several classes are represented
    assert len(unique - {LIDARHD_7CL_IGNORE_INDEX}) >= 3


def test_meta_matches_actual_dataset(built_dataset: Path):
    meta = json.loads((built_dataset / "meta.json").read_text())
    actual_counts = {
        s: sum(1 for _ in (built_dataset / s).iterdir() if _.is_dir())
        for s in ("train", "val", "test")
    }
    assert meta["splits"] == actual_counts


def test_hash_assigner_e2e(tmp_path: Path):
    """The streaming HashSplitAssigner path produces a loadable dataset too."""
    assigner = HashSplitAssigner(train=0.5, val=0.25, test=0.25, seed=7)
    formatter = Ptv3Formatter(feat_keys=("intensity", "normals"))
    writer = Ptv3DatasetWriter(formatter, tmp_path / "ds", assigner)
    # 12 distinct tiles, 1 patch each — large enough to populate all 3 splits
    for i in range(12):
        writer.write_patch(_build_patch(f"htile_{i:03d}", patch_idx=0, n=128, seed=i))
    counts = writer.finalize()

    assert sum(counts.values()) == 12
    populated = {s for s, n in counts.items() if n > 0}
    assert "train" in populated  # train ratio is largest; should never be empty

    # Validate the layout is still loadable by the consumer
    for split in populated:
        ds = MinimalPointceptDataset(tmp_path / "ds", split=split)
        ds.validate()
