"""
Regression tests for the float32 / absolute-coordinate precision bug.

IGN LiDAR HD tiles are stored in Lambert-93 (EPSG:2154): X ≈ 6.5e5 m,
Y ≈ 6.9e6 m. A float32 only resolves ~0.06 m in X and ~0.5 m in Y at that
magnitude, so casting absolute coordinates to float32 anywhere upstream of a
k-NN/PCA computation snaps the cloud onto a coarse grid and stripes every
geometric feature.

These tests pin the three properties that keep that from happening again:
  1. geometric features are invariant to a Lambert-93-sized translation,
  2. `recenter_to_local_f32` round-trips far better than a raw cast,
  3. the DTM path still receives *absolute* coordinates,
  4. the tile loader does not quantise coordinates on load.

Plus the companion fix: `horizontality` is always present in the CPU
strategy output (it used to be missing entirely, although every downstream
consumer expects it).
"""

import numpy as np
import pytest

from ign_lidar.features.compute.architectural import compute_horizontality
from ign_lidar.features.compute.coord_utils import recenter_to_local_f32
from ign_lidar.features.compute.features import compute_all_features_optimized
from ign_lidar.features.compute.height import compute_height_above_ground
from ign_lidar.features.strategy_cpu import CPUStrategy

# Realistic Lambert-93 magnitude (Bourgogne / Île-de-France order)
LAMBERT93_OFFSET = np.array([650_000.0, 6_860_000.0, 0.0])


def _synthetic_cloud(n_points: int = 4000, seed: int = 42) -> np.ndarray:
    """
    Roof-like scene: two intersecting planes (a ridge) plus a vertical wall.

    Mixing planar, linear (the ridge/edge) and vertical structures is what
    makes planarity/linearity/normals sensitive to coordinate quantisation.
    """
    rng = np.random.default_rng(seed)
    n_slope = n_points // 2
    n_wall = n_points - n_slope

    # Two roof pitches meeting on a ridge at x = 5
    xy = rng.uniform(0.0, 10.0, size=(n_slope, 2))
    z = np.where(xy[:, 0] < 5.0, 0.4 * xy[:, 0], 0.4 * (10.0 - xy[:, 0]))
    roof = np.column_stack([xy, z])

    # Vertical facade at y = 0
    wall_x = rng.uniform(0.0, 10.0, n_wall)
    wall_z = rng.uniform(0.0, 3.0, n_wall)
    wall = np.column_stack([wall_x, np.zeros(n_wall), wall_z])

    cloud = np.vstack([roof, wall])
    # Realistic LiDAR noise (~1 cm), well below the float32 quantisation
    # step at Lambert-93 magnitude (0.06-0.5 m) that this test guards against
    cloud += rng.normal(0.0, 0.01, size=cloud.shape)
    return cloud.astype(np.float64)


# ============================================================================
# 1. Translation invariance of the canonical CPU implementation
# ============================================================================


def test_features_invariant_to_lambert93_translation():
    """
    Features must be identical whether points are local or absolute.

    Before the fix, `compute_all_features_optimized` cast the whole array to
    float32 *before* building the KD-tree, so the absolute-coordinate run
    differed by ~0.4-0.5 on average for planarity/linearity.
    """
    local = _synthetic_cloud()
    absolute = local + LAMBERT93_OFFSET

    feats_local = compute_all_features_optimized(
        local, k_neighbors=20, compute_advanced=True
    )
    feats_absolute = compute_all_features_optimized(
        absolute, k_neighbors=20, compute_advanced=True
    )

    for name in (
        "planarity",
        "linearity",
        "sphericity",
        "curvature",
        "anisotropy",
        "verticality",
    ):
        np.testing.assert_allclose(
            feats_absolute[name],
            feats_local[name],
            atol=1e-4,
            err_msg=f"'{name}' is not invariant to a Lambert-93 translation",
        )

    # Normals are sign-ambiguous (eigenvector), compare their orientation
    alignment = np.abs(
        np.sum(feats_local["normals"] * feats_absolute["normals"], axis=1)
    )
    np.testing.assert_allclose(alignment, 1.0, atol=1e-4)

    # Horizontality is derived from the normals, same invariance
    np.testing.assert_allclose(
        compute_horizontality(feats_absolute["normals"]),
        compute_horizontality(feats_local["normals"]),
        atol=1e-4,
    )


def test_compute_all_features_does_not_mutate_caller_array():
    """
    Recentering happens on a copy: the DTM path (and every other consumer)
    keeps reading the caller's absolute float64 coordinates.
    """
    absolute = _synthetic_cloud(n_points=500) + LAMBERT93_OFFSET
    untouched = absolute.copy()

    compute_all_features_optimized(absolute, k_neighbors=20, compute_advanced=False)

    assert absolute.dtype == np.float64
    np.testing.assert_array_equal(absolute, untouched)


def test_features_accept_float32_local_input():
    """Already-local float32 input still works (recentering is idempotent)."""
    local = _synthetic_cloud(n_points=1000).astype(np.float32)
    feats = compute_all_features_optimized(
        local, k_neighbors=20, compute_advanced=False
    )
    assert feats["planarity"].shape == (1000,)
    assert np.all(np.isfinite(feats["planarity"]))


# ============================================================================
# 2. The recentering helper itself
# ============================================================================


def test_recenter_roundtrip_beats_naive_cast():
    """`local + origin` must recover the input far better than a raw cast."""
    rng = np.random.default_rng(0)
    # Full 1 km IGN tile, worst case for the local float32 extent
    absolute = rng.uniform(0.0, 1000.0, size=(10_000, 3)) + LAMBERT93_OFFSET

    local, origin = recenter_to_local_f32(absolute)

    assert local.dtype == np.float32
    assert origin.dtype == np.float64
    assert origin.shape == (3,)

    recentered_err = np.abs(local.astype(np.float64) + origin - absolute).max()
    naive_err = np.abs(absolute.astype(np.float32).astype(np.float64) - absolute).max()

    # float32 relative precision over a 1 km local extent → < 0.1 mm
    assert recentered_err < 1e-3
    # ... versus decimetre-level quantisation for a direct cast
    assert naive_err > 1e-2
    assert recentered_err < naive_err / 1000.0


def test_recenter_patch_sized_extent_is_sub_micrometre():
    """On a patch-sized extent (~50 m) the round-trip is essentially exact."""
    rng = np.random.default_rng(1)
    absolute = rng.uniform(0.0, 50.0, size=(5_000, 3)) + LAMBERT93_OFFSET

    local, origin = recenter_to_local_f32(absolute)

    assert np.abs(local.astype(np.float64) + origin - absolute).max() < 1e-5


def test_recenter_shared_origin():
    """An explicit origin puts two arrays in the same local frame."""
    rng = np.random.default_rng(2)
    reference = rng.uniform(0.0, 100.0, size=(1000, 3)) + LAMBERT93_OFFSET
    query = rng.uniform(0.0, 100.0, size=(50, 3)) + LAMBERT93_OFFSET

    ref_local, origin = recenter_to_local_f32(reference)
    query_local, query_origin = recenter_to_local_f32(query, origin=origin)

    np.testing.assert_array_equal(origin, query_origin)
    # Relative geometry between the two sets is preserved
    np.testing.assert_allclose(
        query_local[0].astype(np.float64) - ref_local[0].astype(np.float64),
        query[0] - reference[0],
        atol=1e-4,
    )


def test_recenter_is_idempotent():
    """Recentering an already-local cloud is harmless."""
    local = _synthetic_cloud(n_points=500)
    once, _ = recenter_to_local_f32(local)
    twice, origin2 = recenter_to_local_f32(once)

    np.testing.assert_allclose(once, twice + origin2.astype(np.float32), atol=1e-5)


def test_recenter_isolates_nan_instead_of_amplifying_it():
    """One non-finite input point must not poison the whole cloud.

    A mean-based origin would turn a single NaN point into a NaN *origin*,
    corrupting every output row instead of just the offending one.
    """
    points = _synthetic_cloud(n_points=200) + LAMBERT93_OFFSET
    points[7, 1] = np.nan

    local, origin = recenter_to_local_f32(points)

    assert np.isfinite(origin).all()
    assert not np.isfinite(local[7]).all()
    finite_rows = np.ones(len(points), dtype=bool)
    finite_rows[7] = False
    assert np.isfinite(local[finite_rows]).all()


def test_recenter_empty_input_does_not_warn_or_produce_nan_origin(recwarn):
    """An empty point array must not trigger a 'mean of empty slice' warning
    nor leave a NaN origin (which would poison a later query recentred
    against it via `origin=`)."""
    empty = np.empty((0, 3), dtype=np.float64)

    local, origin = recenter_to_local_f32(empty)

    assert local.shape == (0, 3)
    assert np.isfinite(origin).all()
    assert not any(
        issubclass(w.category, RuntimeWarning) for w in recwarn.list
    )


# ============================================================================
# 3. DTM offset characterization (no network, stubbed fetcher)
# ============================================================================


class _StubDTMFetcher:
    """Minimal RGEALTIFetcher stand-in: a ground plane sloping with X."""

    def __init__(self):
        self.received_points = None

    def compute_height_above_ground(self, points, crs="EPSG:2154"):
        self.received_points = points.copy()
        ground = self._ground_elevation(points)
        return points[:, 2] - ground

    @staticmethod
    def _ground_elevation(points):
        # Tied to ABSOLUTE X: a bogus (local) query gives a bogus elevation
        return 100.0 + 0.001 * (points[:, 0] - LAMBERT93_OFFSET[0])


def _dtm_test_cloud():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0.0, 100.0, size=(200, 2)) + LAMBERT93_OFFSET[:2]
    ground = 100.0 + 0.001 * (xy[:, 0] - LAMBERT93_OFFSET[0])
    z = ground + rng.uniform(0.0, 15.0, 200)  # 0-15 m above ground
    return np.column_stack([xy, z])


def test_dtm_offset_reconstructs_absolute_coordinates():
    """
    Local points + `dtm_offset` must query the DTM at the same absolute
    location as absolute points with no offset, and yield the same heights.
    """
    absolute = _dtm_test_cloud()
    classification = np.full(len(absolute), 6, dtype=np.uint8)  # building
    offset = np.array([LAMBERT93_OFFSET[0], LAMBERT93_OFFSET[1], 0.0])
    local = absolute - offset

    fetcher_abs = _StubDTMFetcher()
    height_abs = compute_height_above_ground(
        absolute, classification, method="dtm", dtm_fetcher=fetcher_abs
    )

    fetcher_local = _StubDTMFetcher()
    height_local = compute_height_above_ground(
        local, classification, method="dtm",
        dtm_fetcher=fetcher_local, dtm_offset=offset,
    )

    np.testing.assert_allclose(
        fetcher_local.received_points, fetcher_abs.received_points, atol=1e-6
    )
    np.testing.assert_allclose(height_local, height_abs, atol=1e-4)


def test_dtm_offset_does_not_mutate_caller_points():
    """`points + dtm_offset` must not write back into the caller's array."""
    absolute = _dtm_test_cloud()
    offset = np.array([LAMBERT93_OFFSET[0], LAMBERT93_OFFSET[1], 0.0])
    local = absolute - offset
    untouched = local.copy()

    compute_height_above_ground(
        local,
        np.full(len(local), 6, dtype=np.uint8),
        method="dtm",
        dtm_fetcher=_StubDTMFetcher(),
        dtm_offset=offset,
    )

    np.testing.assert_array_equal(local, untouched)


def test_cpu_strategy_forwards_absolute_points_to_dtm():
    """
    The recentering added inside `compute_all_features_optimized` must stay
    invisible to the DTM path: CPUStrategy still hands it the original
    (absolute) coordinates it was called with.
    """
    absolute = _dtm_test_cloud()
    classification = np.full(len(absolute), 6, dtype=np.uint8)
    fetcher = _StubDTMFetcher()

    result = CPUStrategy(k_neighbors=10, verbose=False).compute(
        points=absolute,
        classification=classification,
        height_method="dtm",
        dtm_fetcher=fetcher,
    )

    assert fetcher.received_points is not None
    np.testing.assert_allclose(fetcher.received_points, absolute, atol=1e-9)
    # Heights are 0-15 m above the sloping ground, not ~6.9e6
    assert result["height"].max() < 20.0


# ============================================================================
# 4. Tile loader keeps float64
# ============================================================================


def _write_synthetic_las(tmp_path, n_points=500):
    """Write a tiny uncompressed LAS with Lambert-93 coordinates."""
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(4)
    xyz = rng.uniform(0.0, 1000.0, size=(n_points, 3)) + LAMBERT93_OFFSET

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = rng.integers(0, 65535, n_points).astype(np.uint16)
    las.return_number = np.ones(n_points, dtype=np.uint8)
    las.classification = np.full(n_points, 6, dtype=np.uint8)

    path = tmp_path / "synthetic_tile.las"
    las.write(str(path))
    return path, xyz


def _tile_loader(chunk_size_mb=500):
    from omegaconf import OmegaConf

    from ign_lidar.core.classification.io.tiles import TileLoader

    config = OmegaConf.create({"processor": {"chunk_size_mb": chunk_size_mb}})
    return TileLoader(config)


def test_tile_loader_keeps_float64(tmp_path):
    """Standard loading must not quantise absolute coordinates to float32."""
    path, xyz = _write_synthetic_las(tmp_path)

    tile_data = _tile_loader().load_tile(path)

    assert tile_data is not None
    points = tile_data["points"]
    assert points.dtype == np.float64, (
        "TileLoader must keep absolute Lambert-93 coordinates in float64 - "
        "a float32 cast here quantises them by 0.06-0.5 m"
    )
    # Only the LAS scale (1 mm) may be lost, not the float32 grid
    assert np.abs(points - xyz).max() < 1e-3


def test_tile_loader_chunked_keeps_float64(tmp_path):
    """Same guarantee on the chunked path used for large files."""
    path, xyz = _write_synthetic_las(tmp_path)

    # Call the chunked implementation directly (the file is far below the
    # chunk_size_mb threshold that would normally trigger it)
    tile_data = _tile_loader()._load_tile_chunked(path, max_retries=1)

    assert tile_data is not None
    assert tile_data["points"].dtype == np.float64
    assert np.abs(tile_data["points"] - xyz).max() < 1e-3


# ============================================================================
# 5. horizontality is always exposed
# ============================================================================


@pytest.mark.parametrize("include_extra", [False, True])
def test_cpu_strategy_always_returns_horizontality(include_extra):
    """
    `horizontality` used to be missing from the CPU strategy output entirely,
    although feature_modes/thresholds/classification all expect it.
    """
    points = _synthetic_cloud(n_points=1000) + LAMBERT93_OFFSET

    result = CPUStrategy(
        k_neighbors=20, include_extra=include_extra, verbose=False
    ).compute(points=points)

    assert "horizontality" in result
    horizontality = result["horizontality"]
    assert horizontality.dtype == np.float32
    assert horizontality.shape == (len(points),)
    assert np.all((horizontality >= 0.0) & (horizontality <= 1.0))
    np.testing.assert_allclose(
        horizontality, np.abs(result["normals"][:, 2]), atol=1e-6
    )
