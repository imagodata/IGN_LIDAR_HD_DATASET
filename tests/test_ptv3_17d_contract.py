"""Contract tests for the PTv3 17D export (XYZ + 14 features).

The 17D stack is consumed positionally on the training side (Pointcept configs
index ``feat.npy`` by column), so this file pins BOTH the column order and the
fact that every column carries real signal. Two of them — ``num_returns`` and
``horizontality`` — used to be silently zero-filled by ``Ptv3Formatter``
because nothing in the pipeline ever produced those keys; the
``std(axis=0) > 0`` assertion below is the regression guard for exactly that.

Contract (never reorder):
    0     intensity           (log-scaled for this preset)
    1     return_number
    2     num_returns
    3:6   normals
    6     curvature
    7     horizontality
    8:11  rgb
    11    nir
    12    ndvi
    13    height_above_ground
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from ign_lidar.io.formatters import Ptv3Formatter

# Column contract — order is load-bearing, see module docstring.
CONTRACT_FEAT_KEYS = (
    "intensity",
    "return_number",
    "num_returns",
    "normals",
    "curvature",
    "horizontality",
    "rgb",
    "nir",
    "ndvi",
    "height_above_ground",
)
CONTRACT_FEAT_DIM = 14
PRESET_17D = (
    Path(__file__).resolve().parents[1]
    / "ign_lidar"
    / "configs"
    / "presets"
    / "ptv3_aerial_17d.yaml"
)


# ---------------------------------------------------------------------------
# Fixtures / synthetic inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_orchestrator_cls():
    """Return the *real* FeatureOrchestrator class.

    ``tests/test_orchestrator_facade.py`` installs a MagicMock into
    ``sys.modules['ign_lidar.features.orchestrator']`` at import time and never
    restores it, so in a full-suite run a plain import here would hand back the
    mock. Reload the genuine module for the duration of the test, then put the
    previous entry back so we don't change behaviour for anything after us.
    """
    name = "ign_lidar.features.orchestrator"
    saved = sys.modules.get(name)
    sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name).FeatureOrchestrator
    finally:
        if saved is not None:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)

def _synthetic_las(path: Path, n: int = 400, seed: int = 0):
    """Write a LAS with *distinct* return_number and number_of_returns.

    Distinct on purpose: a loader that reads ``return_number`` twice would
    still pass a "column is not constant" check, so the values must differ.
    """
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(seed)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([650_000.0, 6_860_000.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = rng.uniform(650_000, 650_050, n)
    las.y = rng.uniform(6_860_000, 6_860_050, n)
    las.z = rng.uniform(0.0, 25.0, n)
    las.intensity = rng.integers(0, 65535, n).astype(np.uint16)
    num_returns = rng.integers(1, 6, n)                    # 1..5
    return_number = rng.integers(1, num_returns + 1)       # 1..num_returns
    las.number_of_returns = num_returns
    las.return_number = return_number
    las.classification = rng.choice([2, 6], size=n).astype(np.uint8)
    las.write(str(path))
    return return_number.astype(np.float32), num_returns.astype(np.float32)


def _synthetic_patch(n: int = 256, seed: int = 0) -> dict:
    """A patch dict carrying every key of the 14-feature contract."""
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {
        # Absolute Lambert-93: exercises the float64 recentering in _build_coord.
        "points": np.column_stack([
            rng.uniform(650_000, 650_150, n),
            rng.uniform(6_860_000, 6_860_150, n),
            rng.uniform(120.0, 160.0, n),
        ]),
        "labels": rng.choice([2, 3, 5, 6, 9, 17], size=n).astype(np.int32),
        "intensity": rng.integers(0, 65535, n).astype(np.float32),
        "return_number": rng.integers(1, 6, n).astype(np.float32),
        "num_returns": rng.integers(1, 6, n).astype(np.float32),
        "normals": normals.astype(np.float32),
        "curvature": rng.uniform(0.0, 0.33, n).astype(np.float32),
        "horizontality": rng.uniform(0.0, 1.0, n).astype(np.float32),
        "rgb": rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32),
        "nir": rng.uniform(0.0, 1.0, n).astype(np.float32),
        "ndvi": rng.uniform(-1.0, 1.0, n).astype(np.float32),
        "height_above_ground": rng.uniform(0.0, 30.0, n).astype(np.float32),
        "patch_id": "tile_000_p0000",
        "tile_id": "tile_000",
    }


# ---------------------------------------------------------------------------
# 1. num_returns survives the real load → features → patch chain
# ---------------------------------------------------------------------------

def test_tile_loader_reads_number_of_returns(tmp_path: Path):
    """TileLoader must expose las.number_of_returns under the "num_returns" key."""
    from ign_lidar.core.classification.io.tiles import TileLoader

    expected_rn, expected_nr = _synthetic_las(tmp_path / "tile.laz", n=400)

    loader = TileLoader(OmegaConf.create({"processor": {}, "preprocess": {}}))
    tile_data = loader.load_tile(tmp_path / "tile.laz")

    assert tile_data is not None
    assert "num_returns" in tile_data
    np.testing.assert_array_equal(tile_data["num_returns"], expected_nr)
    np.testing.assert_array_equal(tile_data["return_number"], expected_rn)
    # Not a copy of return_number, and not zero-filled.
    assert not np.array_equal(tile_data["num_returns"], tile_data["return_number"])
    assert tile_data["num_returns"].std() > 0


def test_num_returns_reaches_the_patch_dict(tmp_path: Path, feature_orchestrator_cls):
    """End of the chain: LAS → TileLoader → features → patch dict."""
    from ign_lidar.core.classification.io.tiles import TileLoader
    from ign_lidar.core.classification.patch_extractor import (
        PatchConfig,
        extract_and_augment_patches,
    )

    FeatureOrchestrator = feature_orchestrator_cls
    _, expected_nr = _synthetic_las(tmp_path / "tile.laz", n=600, seed=3)

    loader = TileLoader(OmegaConf.create({"processor": {}, "preprocess": {}}))
    tile_data = loader.load_tile(tmp_path / "tile.laz")

    orchestrator = FeatureOrchestrator(OmegaConf.create({
        "processor": {"use_gpu": False},
        "features": {
            "mode": "lod2",
            "k_neighbors": 10,
            "use_rgb": False,
            "use_infrared": False,
            "compute_ndvi": False,
            "enable_artifact_filtering": False,
        },
    }))
    features = orchestrator.compute_features(tile_data=tile_data)

    for key in ("num_returns", "return_number", "curvature", "horizontality"):
        assert key in features, f"{key!r} missing from the feature dict"

    patches = extract_and_augment_patches(
        points=tile_data["points"],
        features=features,
        labels=tile_data["classification"],
        patch_config=PatchConfig(patch_size=60.0, overlap=0.0, min_points=10),
    )
    assert patches, "no patch extracted from the synthetic tile"

    for patch in patches:
        for key in ("num_returns", "curvature", "horizontality"):
            assert key in patch, f"{key!r} did not reach the patch dict"
            assert len(patch[key]) == len(patch["points"])
        assert patch["num_returns"].std() > 0

    # Every loaded value must still be somewhere in the extracted patches.
    seen = np.unique(np.concatenate([p["num_returns"] for p in patches]))
    assert set(seen).issubset(set(np.unique(expected_nr)))


# ---------------------------------------------------------------------------
# 2. The formatter accepts the full contract
# ---------------------------------------------------------------------------

def test_formatter_accepts_contract_keys():
    """curvature/horizontality must be valid feat_keys (were rejected before)."""
    fmt = Ptv3Formatter(feat_keys=CONTRACT_FEAT_KEYS)
    assert fmt.feat_keys == CONTRACT_FEAT_KEYS


def test_formatter_meta_reports_14_dims(tmp_path: Path):
    fmt = Ptv3Formatter(feat_keys=CONTRACT_FEAT_KEYS, intensity_scaling="log")
    import json

    meta = json.loads(fmt.write_meta(tmp_path).read_text())
    assert meta["feat_dim"] == CONTRACT_FEAT_DIM
    assert meta["feat_keys"] == list(CONTRACT_FEAT_KEYS)
    assert meta["intensity_scaling"] == "log"


# ---------------------------------------------------------------------------
# 3. Full 17D format, written to disk
# ---------------------------------------------------------------------------

@pytest.fixture
def written_17d_patch(tmp_path: Path) -> tuple:
    fmt = Ptv3Formatter(feat_keys=CONTRACT_FEAT_KEYS, intensity_scaling="log")
    patch = _synthetic_patch(n=256, seed=7)
    target = fmt.write_patch(fmt.format_patch(patch), tmp_path / "train")
    return target, patch


def test_17d_arrays_have_the_expected_shapes_and_dtypes(written_17d_patch):
    target, patch = written_17d_patch
    n = len(patch["points"])

    coord = np.load(target / "coord.npy")
    feat = np.load(target / "feat.npy")
    segment = np.load(target / "segment.npy")
    offset = np.load(target / "offset.npy")

    assert coord.shape == (n, 3) and coord.dtype == np.float32
    assert feat.shape == (n, CONTRACT_FEAT_DIM) and feat.dtype == np.float32
    assert segment.shape == (n,) and segment.dtype == np.int16
    assert offset.shape == (3,) and offset.dtype == np.float64

    valid = segment != -1
    assert valid.any()
    assert segment[valid].min() >= 0 and segment[valid].max() < 7


def test_17d_coord_is_centred_and_reversible(written_17d_patch):
    target, patch = written_17d_patch
    coord = np.load(target / "coord.npy")
    offset = np.load(target / "offset.npy")

    assert np.allclose(coord[:, :2].mean(axis=0), 0.0, atol=1e-3)
    assert coord[:, 2].min() == pytest.approx(0.0, abs=1e-4)

    # coord + offset must rebuild the absolute Lambert-93 coordinates.
    np.testing.assert_allclose(
        coord.astype(np.float64) + offset, patch["points"], atol=1e-3
    )


def test_17d_no_column_is_constant_or_null(written_17d_patch):
    """Guard against the silent zero-fill of a whole feature column."""
    target, _ = written_17d_patch
    feat = np.load(target / "feat.npy")

    stds = feat.std(axis=0)
    dead = [i for i, s in enumerate(stds) if s <= 0.0]
    assert not dead, f"constant/zero-filled feat columns: {dead} (stds={stds})"
    # num_returns (2) and horizontality (7) were the two silently zero-filled
    # columns before this fix — assert them by name so a regression is legible.
    assert stds[2] > 0, "column 2 (num_returns) is dead"
    assert stds[7] > 0, "column 7 (horizontality) is dead"


def test_17d_has_no_nan_or_inf(written_17d_patch):
    target, _ = written_17d_patch
    assert np.isfinite(np.load(target / "feat.npy")).all()
    assert np.isfinite(np.load(target / "coord.npy")).all()


def test_17d_feat_layout_matches_the_contract():
    fmt = Ptv3Formatter(feat_keys=CONTRACT_FEAT_KEYS, intensity_scaling="log")
    formatted = fmt.format_patch(_synthetic_patch(n=64, seed=1))
    assert formatted["feat_layout"] == [
        "intensity",
        "return_number",
        "num_returns",
        "normal_x",
        "normal_y",
        "normal_z",
        "curvature",
        "horizontality",
        "r",
        "g",
        "b",
        "nir",
        "ndvi",
        "height_above_ground",
    ]


# ---------------------------------------------------------------------------
# 4. intensity_scaling: opt-in log, linear stays byte-identical
# ---------------------------------------------------------------------------

def test_intensity_scaling_defaults_to_linear():
    assert Ptv3Formatter().intensity_scaling == "linear"


def test_intensity_linear_is_unchanged():
    """Non-regression: the 9D/12D datasets were built with arr / 65535."""
    fmt = Ptv3Formatter(feat_keys=("intensity",))
    raw = np.array([[0.0], [1.0], [1000.0], [65535.0]], dtype=np.float32)
    patch = {
        "points": np.zeros((4, 3)),
        "labels": np.full(4, 6, dtype=np.int32),
        "intensity": raw.ravel(),
    }
    np.testing.assert_allclose(
        fmt.format_patch(patch)["feat"].ravel(),
        (raw / 65535.0).ravel(),
        rtol=0,
        atol=0,
    )


def test_intensity_log_uses_log1p():
    fmt = Ptv3Formatter(feat_keys=("intensity",), intensity_scaling="log")
    raw = np.array([0.0, 1.0, 1000.0, 65535.0], dtype=np.float32)
    patch = {
        "points": np.zeros((4, 3)),
        "labels": np.full(4, 6, dtype=np.int32),
        "intensity": raw,
    }
    np.testing.assert_allclose(
        fmt.format_patch(patch)["feat"].ravel(),
        np.log1p(raw) / np.log1p(65535.0),
        rtol=1e-6,
    )


def test_intensity_scaling_is_validated():
    with pytest.raises(ValueError):
        Ptv3Formatter(intensity_scaling="sqrt")


# ---------------------------------------------------------------------------
# 5. The 17D preset itself
# ---------------------------------------------------------------------------

def test_preset_17d_declares_the_contract():
    preset = OmegaConf.load(PRESET_17D)
    ptv3_cfg = preset.output.ptv3

    assert list(ptv3_cfg.feat_keys) == list(CONTRACT_FEAT_KEYS)
    assert ptv3_cfg.intensity_scaling == "log"

    fmt = Ptv3Formatter(
        feat_keys=tuple(ptv3_cfg.feat_keys),
        intensity_scaling=str(ptv3_cfg.intensity_scaling),
    )
    formatted = fmt.format_patch(_synthetic_patch(n=32, seed=2))
    assert formatted["feat"].shape[1] == CONTRACT_FEAT_DIM


def test_preset_17d_resolves_to_the_ptv3_writer_through_the_real_loader():
    """Load the preset through HydraRunner (the actual production entry point),
    not a hand-built OmegaConf dict.

    Regression guard: processor.output_format defaults to "laz" in base.yaml and
    silently wins over output.format if a preset forgets to override it — the
    preset would then emit plain LAZ patches and never touch Ptv3Formatter.
    ptv3_aerial.yaml documents this trap explicitly; this test proves the 17D
    preset does not fall into it.
    """
    from ign_lidar.cli.hydra_runner import HydraRunner

    cfg = HydraRunner().load_config(config_file=str(PRESET_17D))

    assert cfg.processor.output_format == "ptv3_pointcept"
    assert cfg.processor.get("enable_optimizations", True) is False or (
        OmegaConf.select(cfg, "enable_optimizations") is False
    )
    assert list(cfg.output.ptv3.feat_keys) == list(CONTRACT_FEAT_KEYS)


def test_output_writer_honours_the_preset_intensity_scaling(tmp_path: Path):
    """OutputWriter must forward output.ptv3.intensity_scaling to the formatter.

    This is the only place where the YAML meets the formatter in production;
    without the wiring the 17D dataset would silently ship linear intensity.
    """
    import json

    from ign_lidar.core.output_writer import PTV3_FORMAT, OutputWriter

    ptv3_cfg = OmegaConf.load(PRESET_17D).output.ptv3
    writer = OutputWriter(OmegaConf.create({
        "processor": {
            "output_format": PTV3_FORMAT,
            "processing_mode": "patches_only",
            "architecture": "ptv3",
            "lod_level": "LOD2",
            "patch_size": 150.0,
        },
        "output": {"ptv3": ptv3_cfg},
    }))

    laz_file = tmp_path / "LHD_FXX_0942_6543_2024.laz"
    laz_file.touch()
    patch = _synthetic_patch(n=64, seed=11)
    patch.update({"_patch_idx": 0, "_version": "original"})
    raw_intensity = patch["intensity"].copy()

    writer.save_patches([patch], laz_file, tmp_path / "out")
    writer.finalize()

    ds_root = tmp_path / "out" / "ptv3_dataset"
    meta = json.loads((ds_root / "meta.json").read_text())
    assert meta["feat_dim"] == CONTRACT_FEAT_DIM
    assert meta["intensity_scaling"] == "log"

    sample = next(
        d
        for split in ("train", "val", "test")
        if (ds_root / split).exists()
        for d in (ds_root / split).iterdir()
        if d.is_dir()
    )
    feat = np.load(sample / "feat.npy")
    assert feat.shape == (64, CONTRACT_FEAT_DIM)
    np.testing.assert_allclose(
        feat[:, 0], np.log1p(raw_intensity) / np.log1p(65535.0), rtol=1e-6
    )
    assert (feat.std(axis=0) > 0).all()


def test_existing_presets_are_untouched():
    """9D/12D presets must keep linear intensity (no silent scaling change)."""
    presets_dir = PRESET_17D.parent
    for name in ("ptv3_aerial.yaml", "ptv3_aerial_12d.yaml"):
        ptv3_cfg = OmegaConf.load(presets_dir / name).output.ptv3
        assert "intensity_scaling" not in ptv3_cfg
        fmt = Ptv3Formatter(feat_keys=tuple(ptv3_cfg.feat_keys))
        assert fmt.intensity_scaling == "linear"
