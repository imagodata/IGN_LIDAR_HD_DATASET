"""
Unit tests for core.height module.

Tests canonical height computation implementations.
"""

import numpy as np
import pytest
from ign_lidar.features.core.height import (
    compute_height_above_ground,
    compute_relative_height,
    compute_normalized_height,
    compute_height_percentile,
    compute_height_bins,
)


class TestComputeHeightAboveGround:
    """Tests for compute_height_above_ground function."""
    
    def test_basic_ground_plane(self):
        """Test basic height computation with ground plane method."""
        # Create simple point cloud with known ground and elevated points
        points = np.array([
            [0, 0, 0],    # ground
            [1, 1, 0],    # ground
            [2, 2, 0],    # ground
            [3, 3, 5],    # elevated
            [4, 4, 10],   # elevated
        ], dtype=np.float32)
        
        classification = np.array([2, 2, 2, 6, 6])  # 2=ground, 6=building
        
        height = compute_height_above_ground(points, classification, method='ground_plane')
        
        expected = np.array([0, 0, 0, 5, 10], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)
    
    def test_min_z_method(self):
        """Test height computation with min_z method."""
        points = np.array([
            [0, 0, 5],
            [1, 1, 10],
            [2, 2, 15],
        ], dtype=np.float32)
        
        classification = np.array([1, 1, 1])
        
        height = compute_height_above_ground(points, classification, method='min_z')
        
        expected = np.array([0, 5, 10], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)
    
    def test_no_ground_points_fallback(self):
        """Test fallback to min_z when no ground points found."""
        points = np.array([
            [0, 0, 10],
            [1, 1, 20],
            [2, 2, 30],
        ], dtype=np.float32)
        
        classification = np.array([6, 6, 6])  # all buildings, no ground
        
        # Should fallback to global min Z
        height = compute_height_above_ground(points, classification, method='ground_plane')
        
        expected = np.array([0, 10, 20], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)
    
    def test_custom_ground_class(self):
        """Test using custom ground class."""
        points = np.array([
            [0, 0, 0],    # water
            [1, 1, 5],    # building
            [2, 2, 10],   # building
        ], dtype=np.float32)
        
        classification = np.array([9, 6, 6])  # 9=water, 6=building
        
        # Use water as reference
        height = compute_height_above_ground(
            points, classification, 
            method='ground_plane', 
            ground_class=9
        )
        
        expected = np.array([0, 5, 10], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)
    
    def test_negative_heights_clamped(self):
        """Test that negative heights are clamped to zero."""
        points = np.array([
            [0, 0, 10],   # ground reference
            [1, 1, 5],    # below ground
            [2, 2, 15],   # above ground
        ], dtype=np.float32)
        
        classification = np.array([2, 1, 1])
        
        height = compute_height_above_ground(points, classification)
        
        # Below-ground point should be clamped to 0
        assert height[1] == 0.0
        assert height[0] == 0.0
        assert height[2] == 5.0
    
    def test_dtm_method_uses_fetcher(self):
        """DTM method samples ground elevation from the given fetcher."""
        points = np.array([
            [0, 0, 10],
            [1, 1, 8],
        ], dtype=np.float32)
        classification = np.array([2, 6])

        class FakeDTMFetcher:
            def compute_height_above_ground(self, points, dtm_data=None, bbox=None, crs="EPSG:2154"):
                # Flat terrain at elevation 5m
                return points[:, 2] - 5.0

        height = compute_height_above_ground(
            points, classification, method='dtm', dtm_fetcher=FakeDTMFetcher()
        )
        np.testing.assert_allclose(height, [5.0, 3.0])

    def test_dtm_method_falls_back_on_fetch_failure(self):
        """DTM method falls back to ground_plane if the fetcher returns None."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 5],
        ], dtype=np.float32)
        classification = np.array([2, 6])

        class FailingDTMFetcher:
            def compute_height_above_ground(self, points, dtm_data=None, bbox=None, crs="EPSG:2154"):
                return None

        height = compute_height_above_ground(
            points, classification, method='dtm', dtm_fetcher=FailingDTMFetcher()
        )
        expected = compute_height_above_ground(points, classification, method='ground_plane')
        np.testing.assert_allclose(height, expected)

    def test_dtm_offset_shifts_points_before_fetch(self):
        """dtm_offset must reach the fetcher as absolute coords, and the
        returned height must reflect the absolute Z, not the local one
        (recentered patches, e.g. FRACTAL's coord_absolute = coord + offset).
        """
        # Patch-local coordinates: small, centered near zero.
        local_points = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        classification = np.array([6, 2])
        offset = np.array([650000.0, 6860000.0, 100.0])

        seen = {}

        class RecordingDTMFetcher:
            def compute_height_above_ground(self, points, dtm_data=None, bbox=None, crs="EPSG:2154"):
                seen["points"] = points.copy()
                # Flat terrain at absolute elevation 95m.
                return points[:, 2] - 95.0

        height = compute_height_above_ground(
            local_points, classification, method='dtm',
            dtm_fetcher=RecordingDTMFetcher(), dtm_offset=offset,
        )
        # Fetcher must have seen absolute coordinates (local + offset).
        np.testing.assert_allclose(seen["points"][:, 0], [650000.0, 650001.0])
        np.testing.assert_allclose(seen["points"][:, 1], [6860000.0, 6860001.0])
        np.testing.assert_allclose(seen["points"][:, 2], [102.0, 100.0])
        # Height = absolute_z - 95 = (local_z + 100) - 95 = local_z + 5.
        np.testing.assert_allclose(height, [7.0, 5.0])
    
    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        classification = np.array([2])
        
        with pytest.raises(ValueError, match="Unknown method"):
            compute_height_above_ground(points, classification, method='invalid')
    
    def test_invalid_points_shape(self):
        """Test validation of points array shape."""
        with pytest.raises(ValueError, match="must have shape"):
            compute_height_above_ground(
                np.array([0, 0, 0]),  # 1D instead of 2D
                np.array([2])
            )
    
    def test_invalid_classification_shape(self):
        """Test validation of classification array shape."""
        with pytest.raises(ValueError, match="must be 1D array"):
            compute_height_above_ground(
                np.array([[0, 0, 0]]),
                np.array([[2]])  # 2D instead of 1D
            )
    
    def test_mismatched_lengths(self):
        """Test validation of matching array lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            compute_height_above_ground(
                np.array([[0, 0, 0], [1, 1, 1]]),
                np.array([2])  # length mismatch
            )
    
    def test_output_dtype(self):
        """Test that output is float32."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        classification = np.array([2])
        
        height = compute_height_above_ground(points, classification)
        
        assert height.dtype == np.float32


class TestComputeRelativeHeight:
    """Tests for compute_relative_height function."""
    
    def test_relative_to_ground(self):
        """Test relative height with default ground reference."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 5],
        ], dtype=np.float32)
        
        classification = np.array([2, 6])
        
        height = compute_relative_height(points, classification, reference_class=2)
        
        expected = np.array([0, 5], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)
    
    def test_relative_to_water(self):
        """Test relative height with water reference."""
        points = np.array([
            [0, 0, 0],    # water
            [1, 1, 10],   # building
        ], dtype=np.float32)
        
        classification = np.array([9, 6])  # 9=water
        
        height = compute_relative_height(points, classification, reference_class=9)
        
        expected = np.array([0, 10], dtype=np.float32)
        np.testing.assert_array_almost_equal(height, expected)


class TestComputeNormalizedHeight:
    """Tests for compute_normalized_height function."""
    
    def test_auto_max_height(self):
        """Test normalized height with automatic max detection."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 5],
            [2, 2, 10],
        ], dtype=np.float32)
        
        classification = np.array([2, 6, 6])
        
        height_norm = compute_normalized_height(points, classification)
        
        expected = np.array([0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(height_norm, expected)
    
    def test_fixed_max_height(self):
        """Test normalized height with fixed maximum."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 25],
            [2, 2, 50],
        ], dtype=np.float32)
        
        classification = np.array([2, 6, 6])
        
        height_norm = compute_normalized_height(points, classification, max_height=50.0)
        
        expected = np.array([0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(height_norm, expected)
    
    def test_exceeding_max_height_clamped(self):
        """Test that heights exceeding max are clamped to 1.0."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 60],   # exceeds max
        ], dtype=np.float32)
        
        classification = np.array([2, 6])
        
        height_norm = compute_normalized_height(points, classification, max_height=50.0)
        
        assert height_norm[0] == 0.0
        assert height_norm[1] == 1.0  # clamped
    
    def test_zero_max_height(self):
        """Test behavior when max height is zero."""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        classification = np.array([2])
        
        height_norm = compute_normalized_height(points, classification)
        
        assert height_norm[0] == 0.0


class TestComputeHeightPercentile:
    """Tests for compute_height_percentile function."""
    
    def test_95th_percentile(self):
        """Test 95th percentile computation."""
        # Create heights: 0, 1, 2, ..., 99
        points = np.column_stack([
            np.zeros(100),
            np.zeros(100),
            np.arange(100, dtype=np.float32)
        ])
        
        classification = np.array([2] + [6] * 99)
        
        h95 = compute_height_percentile(points, classification, percentile=95.0)
        
        # 95th percentile of 0..99 should be around 95
        assert 94 <= h95 <= 96
    
    def test_median_percentile(self):
        """Test median (50th percentile) computation."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 10],
            [2, 2, 20],
        ], dtype=np.float32)
        
        classification = np.array([2, 6, 6])
        
        h50 = compute_height_percentile(points, classification, percentile=50.0)
        
        # Median of [0, 10, 20] is 10
        assert h50 == 10.0


class TestComputeHeightBins:
    """Tests for compute_height_bins function."""
    
    def test_evenly_spaced_bins(self):
        """Test evenly spaced bin creation."""
        points = np.array([
            [0, 0, 0],
            [1, 1, 5],
            [2, 2, 10],
        ], dtype=np.float32)
        
        classification = np.array([2, 6, 6])
        
        bins = compute_height_bins(points, classification, num_bins=2)
        
        # Heights: 0, 5, 10 -> bins: 0-5, 5-10
        # Expected: bin 0, bin 0/1, bin 1
        assert bins[0] == 0  # height 0 -> bin 0
        assert bins[2] == 1  # height 10 -> bin 1
    
    def test_custom_bin_edges(self):
        """Test custom bin edges."""
        points = np.array([
            [0, 0, 0],     # 0m: ground layer
            [1, 1, 1],     # 1m: understory
            [2, 2, 3],     # 3m: mid canopy
            [3, 3, 10],    # 10m: canopy
            [4, 4, 25],    # 25m: emergent
        ], dtype=np.float32)
        
        classification = np.array([2, 3, 3, 5, 5])
        
        # Vegetation layers: 0-0.5, 0.5-2, 2-5, 5-15, 15-30
        layer_edges = np.array([0, 0.5, 2.0, 5.0, 15.0, 30.0])
        
        bins = compute_height_bins(points, classification, bin_edges=layer_edges)
        
        # Verify layers
        assert bins[0] == 0  # 0m -> layer 0 (0-0.5)
        assert bins[1] == 1  # 1m -> layer 1 (0.5-2.0)
        assert bins[2] == 2  # 3m -> layer 2 (2.0-5.0)
        assert bins[3] == 3  # 10m -> layer 3 (5.0-15.0)
        assert bins[4] == 4  # 25m -> layer 4 (15.0-30.0)
    
    def test_output_dtype(self):
        """Test that output is int32."""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        classification = np.array([2])
        
        bins = compute_height_bins(points, classification, num_bins=5)
        
        assert bins.dtype == np.int32


class TestIntegration:
    """Integration tests combining multiple height functions."""
    
    def test_workflow_building_analysis(self):
        """Test typical workflow for building height analysis."""
        # Simulate building point cloud
        np.random.seed(42)
        n_ground = 100
        n_building = 50
        
        # Ground points at z=0
        ground_points = np.column_stack([
            np.random.rand(n_ground) * 10,
            np.random.rand(n_ground) * 10,
            np.zeros(n_ground)
        ])
        
        # Building points at various heights
        building_points = np.column_stack([
            np.random.rand(n_building) * 10,
            np.random.rand(n_building) * 10,
            np.random.rand(n_building) * 20 + 5  # 5-25m
        ])
        
        points = np.vstack([ground_points, building_points]).astype(np.float32)
        classification = np.array([2] * n_ground + [6] * n_building)
        
        # Compute various height metrics
        height = compute_height_above_ground(points, classification)
        height_norm = compute_normalized_height(points, classification, max_height=30.0)
        h95 = compute_height_percentile(points, classification, percentile=95.0)
        bins = compute_height_bins(points, classification, num_bins=5)
        
        # Verify results
        assert len(height) == len(points)
        assert np.all(height >= 0)
        assert np.all((height_norm >= 0) & (height_norm <= 1))
        assert 0 < h95 < 30
        assert np.all((bins >= 0) & (bins <= 4))
        
        # Ground points should have zero height
        assert np.all(height[:n_ground] == 0)
        
        # Building points should have positive height
        assert np.all(height[n_ground:] > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
