"""Regression test: features dict keys colliding with native LAS dimensions.

`save_enriched_tile_laz()` is commonly called with a `features` dict coming
straight from `FeatureOrchestrator.compute_features()`, which includes
'red'/'green'/'blue'/'nir' alongside geometric features. When RGB/NIR are
also supplied via `input_rgb`/`input_nir` (point format 7/8), those keys
collide with native LAS dimensions: `add_extra_dim()` raises
"field '<name>' occurs more than once" AND leaves the LasData point record
in a state where every *subsequent* add_extra_dim() call also fails,
regardless of name — so a single colliding key silently dropped nearly
every other computed feature (ndvi, height_above_ground, is_ground, ...)
from the output, not just the colliding one.
"""

from pathlib import Path

import laspy
import numpy as np
import pytest

from ign_lidar.core.classification.io import save_enriched_tile_laz


@pytest.fixture
def sample_data():
    n = 200
    rng = np.random.default_rng(0)
    return {
        "points": np.column_stack([
            rng.uniform(650000, 650050, n),
            rng.uniform(6860000, 6860050, n),
            rng.uniform(100, 120, n),
        ]),
        "classification": rng.choice([2, 6], n).astype(np.uint8),
        "intensity": rng.uniform(0, 1, n).astype(np.float32),
        "return_number": np.ones(n, dtype=np.uint8),
        "rgb": rng.uniform(0, 1, (n, 3)).astype(np.float32),
        "nir": rng.uniform(0, 1, n).astype(np.float32),
    }


def test_colliding_feature_keys_do_not_break_other_features(sample_data, tmp_path):
    n = len(sample_data["points"])
    features = {
        "normals": np.random.uniform(-1, 1, (n, 3)).astype(np.float32),
        "height_above_ground": np.random.uniform(0, 20, n).astype(np.float32),
        "ndvi": np.random.uniform(-1, 1, n).astype(np.float32),
        "is_ground": (sample_data["classification"] == 2).astype(np.uint8),
        # These collide with native point-format dimensions once
        # input_rgb/input_nir are provided:
        "nir": sample_data["nir"],
        "red": sample_data["rgb"][:, 0],
        "green": sample_data["rgb"][:, 1],
        "blue": sample_data["rgb"][:, 2],
        "curvature": np.random.uniform(0, 1, n).astype(np.float32),
    }

    save_path = tmp_path / "patch_enriched.laz"
    save_enriched_tile_laz(
        save_path=save_path,
        points=sample_data["points"],
        classification=sample_data["classification"],
        intensity=sample_data["intensity"],
        return_number=sample_data["return_number"],
        features=features,
        input_rgb=sample_data["rgb"],
        input_nir=sample_data["nir"],
    )

    las = laspy.read(save_path)
    dims = set(las.point_format.dimension_names)
    for expected in [
        "height_above_ground", "ndvi", "is_ground", "curvature",
        "normal_x", "normal_y", "normal_z",
    ]:
        assert expected in dims, f"'{expected}' missing — cascading failure regression"

    # Native fields still hold the real RGB/NIR data, not garbage.
    np.testing.assert_allclose(
        np.asarray(las.nir) / 65535.0, sample_data["nir"], atol=1e-3
    )
