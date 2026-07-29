"""Regression tests for RGEALTIFetcher DTM sampling.

Covers a bug found running a full FRACTAL conversion at scale: under
concurrent load, IGN's WMS sometimes returns a "successful" (HTTP 200)
response containing a blank/all-nodata GeoTIFF. Before the fix, this
produced a bogus elevation (the nodata sentinel, e.g. -99999.0) treated as
real, giving height_above_ground = z - (-99999) — a huge value that got
silently clipped to a constant 1.0 during feature scaling. 88% of patches
in a live run were affected.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ign_lidar.io.rge_alti_fetcher import RGEALTIFetcher


def _make_geotiff_bytes(grid_value, nodata=-99999.0, shape=(50, 50)):
    grid = np.full(shape, grid_value, dtype=np.float32)
    transform = from_bounds(650000, 6860000, 650050, 6860050, shape[1], shape[0])
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = tmp.name
    try:
        with rasterio.open(
            path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
            dtype=grid.dtype, crs="EPSG:2154", transform=transform, nodata=nodata,
        ) as dst:
            dst.write(grid, 1)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


@pytest.fixture
def fetcher():
    return RGEALTIFetcher(cache_dir=None)


@pytest.fixture
def points():
    n = 100
    rng = np.random.default_rng(0)
    return np.column_stack([
        rng.uniform(650000, 650050, n),
        rng.uniform(6860000, 6860050, n),
        rng.uniform(100, 120, n),
    ])


def _dtm_data(grid_value, nodata=-99999.0, shape=(50, 50)):
    grid = np.full(shape, grid_value, dtype=np.float32)
    transform = from_bounds(650000, 6860000, 650050, 6860050, shape[1], shape[0])
    return grid, {"transform": transform, "nodata": nodata}


class TestAllNodataGrid:
    def test_sample_elevation_returns_none_not_bogus_value(self, fetcher, points):
        """An all-nodata grid must yield None, never the nodata sentinel
        treated as a real elevation."""
        dtm_data = _dtm_data(grid_value=-99999.0, nodata=-99999.0)
        result = fetcher.sample_elevation_at_points(points, dtm_data=dtm_data)
        assert result is None

    def test_compute_height_above_ground_returns_none(self, fetcher, points):
        dtm_data = _dtm_data(grid_value=-99999.0, nodata=-99999.0)
        result = fetcher.compute_height_above_ground(points, dtm_data=dtm_data)
        assert result is None


class TestPartialNodataGrid:
    def test_valid_grid_still_works(self, fetcher, points):
        """Sanity check: a fully valid flat grid still computes correctly."""
        dtm_data = _dtm_data(grid_value=95.0, nodata=-99999.0)
        result = fetcher.sample_elevation_at_points(points, dtm_data=dtm_data)
        assert result is not None
        np.testing.assert_allclose(result, 95.0)

    def test_isolated_nodata_pixels_are_interpolated(self, fetcher, points):
        """Most of the grid valid, a few nodata pixels: nearest-neighbor
        interpolation should recover a value, not fail the whole patch."""
        grid, meta = _dtm_data(grid_value=95.0, nodata=-99999.0)
        grid[0, 0] = -99999.0  # a single isolated nodata pixel, corner
        result = fetcher.sample_elevation_at_points(points, dtm_data=(grid, meta))
        assert result is not None
        assert np.all(result != -99999.0)


class TestBlankWmsResponseRetried:
    def test_blank_then_valid_response_recovers_real_data(self, fetcher):
        """A blank (all-nodata) WMS response is retried on the same layer;
        a subsequent valid response must be used instead of giving up."""
        blank_bytes = _make_geotiff_bytes(-99999.0)
        valid_bytes = _make_geotiff_bytes(95.0)
        calls = {"n": 0}

        def fake_get_with_retry(url, params, timeout=60, operation_name=""):
            calls["n"] += 1
            resp = MagicMock()
            resp.headers = {"Content-Type": "image/geotiff"}
            resp.content = blank_bytes if calls["n"] == 1 else valid_bytes
            return resp

        with patch(
            "ign_lidar.utils.http_retry.get_with_retry", side_effect=fake_get_with_retry
        ):
            result = fetcher._fetch_from_wms(
                (650000, 6860000, 650050, 6860050), "EPSG:2154"
            )

        assert calls["n"] == 2, "must retry the same layer on a blank response"
        assert result is not None
        grid, meta = result
        assert grid[0, 0] == 95.0

    def test_persistently_blank_response_falls_through_to_none(self, fetcher):
        """If every attempt (both layers, all retries) comes back blank,
        _fetch_from_wms must give up cleanly (None), not return garbage."""
        blank_bytes = _make_geotiff_bytes(-99999.0)

        def fake_get_with_retry(url, params, timeout=60, operation_name=""):
            resp = MagicMock()
            resp.headers = {"Content-Type": "image/geotiff"}
            resp.content = blank_bytes
            return resp

        with patch(
            "ign_lidar.utils.http_retry.get_with_retry", side_effect=fake_get_with_retry
        ):
            result = fetcher._fetch_from_wms(
                (650000, 6860000, 650050, 6860050), "EPSG:2154"
            )

        assert result is None
