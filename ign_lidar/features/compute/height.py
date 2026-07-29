"""
Canonical implementation of height-based features.

This module provides height computation replacing duplicates in:
- features.py (CPU)
- features_gpu.py (GPU)
- features_gpu_chunked.py (GPU chunked)
"""

import numpy as np
from typing import Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)


def compute_height_above_ground(
    points: np.ndarray,
    classification: np.ndarray,
    method: Literal['ground_plane', 'min_z', 'dtm'] = 'ground_plane',
    ground_class: int = 2,
    dtm_fetcher: Optional[Any] = None,
    crs: str = "EPSG:2154",
    dtm_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute height above ground for each point.
    
    This is the canonical implementation used across all feature computation
    strategies (CPU, GPU, GPU-chunked).
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    classification : np.ndarray
        ASPRS classification codes of shape (N,)
    method : str, optional
        Height computation method:
        - 'ground_plane': Use minimum Z of ground points (ASPRS class 2)
        - 'min_z': Use global minimum Z
        - 'dtm': Reserved for future DTM-based computation
        (default: 'ground_plane')
    ground_class : int, optional
        ASPRS classification code for ground points (default: 2)
    dtm_fetcher : RGEALTIFetcher, optional
        Pre-instantiated DTM fetcher to reuse across tiles (avoids re-opening
        the cache/local files per call). Only used when method='dtm'. If None,
        a default RGEALTIFetcher() is created on the fly.
    crs : str, optional
        Coordinate reference system of `points` (default: "EPSG:2154",
        Lambert-93). Only used when method='dtm'.
    dtm_offset : np.ndarray, optional
        (3,) offset added to `points` before DTM bbox derivation/sampling
        only (method='dtm'). Height above ground is a physical quantity
        (z_absolute - ground_elevation) independent of any coordinate
        convention, so callers that pass already-recentered/local `points`
        (e.g. `coord_absolute = coord + offset`, as produced by patch
        formatters) must supply that same `offset` here — otherwise the DTM
        fetch queries a bogus bbox near the coordinate origin instead of the
        real patch location. None (default) assumes `points` are already
        absolute Lambert-93 coordinates (unchanged behavior).

    Returns
    -------
    height : np.ndarray
        Height above ground in meters, shape (N,)
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid parameters
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3) * 10
    >>> classification = np.random.choice([1, 2, 3], 1000)
    >>> height = compute_height_above_ground(points, classification)
    >>> assert height.shape == (1000,)
    >>> assert np.all(height >= 0)
    
    >>> # Use different ground class (e.g., low vegetation)
    >>> height = compute_height_above_ground(points, classification, ground_class=3)
    
    Notes
    -----
    Height computation methods:
    
    - **ground_plane**: Uses minimum Z of points with specified ground class.
      If no ground points found, falls back to global minimum Z.
      This is the recommended method for most use cases.
      
    - **min_z**: Uses global minimum Z as ground reference.
      Simpler but may be inaccurate if scene has varying terrain.
      
    - **dtm**: Uses an interpolated ground surface sampled from IGN's
      RGE ALTI®/LiDAR HD MNT digital terrain model (via `RGEALTIFetcher`,
      see `ign_lidar.io.rge_alti_fetcher`), instead of a single per-tile/
      per-patch scalar. Recommended on sloped/mountainous terrain, where
      `ground_plane` over- or under-estimates height depending on where in
      the tile the (few) classified ground points happen to sit. Falls back
      to `ground_plane` with a warning if the DTM fetch fails (e.g. offline,
      WMS unavailable, no local tiles).
    """
    # Input validation
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not isinstance(classification, np.ndarray) or classification.ndim != 1:
        raise ValueError(f"classification must be 1D array, got shape {classification.shape}")
    if len(points) != len(classification):
        raise ValueError(f"points and classification must have same length: "
                       f"{len(points)} != {len(classification)}")
    
    if method == 'ground_plane':
        # Use minimum Z of ground points
        ground_mask = (classification == ground_class)
        if not np.any(ground_mask):
            logger.warning(f"No ground points (class {ground_class}) found, using global min Z")
            ground_z = np.min(points[:, 2])
        else:
            ground_z = np.min(points[ground_mask, 2])
    
    elif method == 'min_z':
        # Use global minimum Z
        ground_z = np.min(points[:, 2])

    elif method == 'dtm':
        # Per-point ground elevation from RGE ALTI/LiDAR HD MNT, not a single
        # scalar — bypasses the shared points[:, 2] - ground_z tail below.
        if dtm_fetcher is None:
            from ...io.rge_alti_fetcher import RGEALTIFetcher
            dtm_fetcher = RGEALTIFetcher()

        # Height above ground is a physical quantity (z_absolute - DTM
        # elevation) — the DTM fetcher needs absolute Lambert-93 coordinates
        # for both bbox derivation and per-point raster sampling, so shift
        # already-recentered `points` back to absolute here if an offset was
        # supplied (see dtm_offset docstring above).
        dtm_points = points if dtm_offset is None else points + dtm_offset

        dtm_ground_z = dtm_fetcher.compute_height_above_ground(dtm_points, crs=crs)
        if dtm_ground_z is None:
            logger.warning(
                "DTM height computation failed (fetch/read error) — "
                "falling back to method='ground_plane'"
            )
            return compute_height_above_ground(
                points, classification, method='ground_plane',
                ground_class=ground_class,
            )

        height = np.maximum(dtm_ground_z, 0.0)
        return height.astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method}. "
                       f"Choose from: 'ground_plane', 'min_z', 'dtm'")
    
    # Compute height and ensure non-negative
    height = points[:, 2] - ground_z
    height = np.maximum(height, 0.0)
    
    return height.astype(np.float32)


def compute_relative_height(
    points: np.ndarray,
    classification: np.ndarray,
    reference_class: int = 2
) -> np.ndarray:
    """
    Compute relative height with respect to a reference class.
    
    This is an alias for compute_height_above_ground() for backward compatibility
    and semantic clarity when using non-ground reference classes.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    classification : np.ndarray
        ASPRS classification codes
    reference_class : int, optional
        Reference class for height computation (default: 2 = ground)
        Common values:
        - 2: Ground
        - 3: Low vegetation
        - 9: Water
        
    Returns
    -------
    relative_height : np.ndarray
        Relative height in meters
        
    Examples
    --------
    >>> # Height above ground
    >>> height_ground = compute_relative_height(points, classification, reference_class=2)
    
    >>> # Height above water surface
    >>> height_water = compute_relative_height(points, classification, reference_class=9)
    """
    return compute_height_above_ground(
        points, 
        classification, 
        method='ground_plane',
        ground_class=reference_class
    )


def compute_normalized_height(
    points: np.ndarray,
    classification: np.ndarray,
    max_height: Optional[float] = None
) -> np.ndarray:
    """
    Compute normalized height in range [0, 1].
    
    Useful for visualization and as input feature for machine learning.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    classification : np.ndarray
        ASPRS classification codes
    max_height : float, optional
        Maximum height for normalization. If None, uses maximum height in data.
        
    Returns
    -------
    normalized_height : np.ndarray
        Normalized height in range [0, 1]
        
    Examples
    --------
    >>> height_norm = compute_normalized_height(points, classification)
    >>> assert np.all((height_norm >= 0) & (height_norm <= 1))
    
    >>> # Normalize with fixed maximum (e.g., 50m for buildings)
    >>> height_norm = compute_normalized_height(points, classification, max_height=50.0)
    """
    height = compute_height_above_ground(points, classification)
    
    if max_height is None:
        max_height = np.max(height)
    
    if max_height > 0:
        normalized = height / max_height
    else:
        normalized = np.zeros_like(height)
    
    # Clamp to [0, 1] in case some points exceed max_height
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def compute_height_percentile(
    points: np.ndarray,
    classification: np.ndarray,
    percentile: float = 95.0
) -> float:
    """
    Compute height percentile of point cloud.
    
    Useful for characterizing overall structure (e.g., canopy height).
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    classification : np.ndarray
        ASPRS classification codes
    percentile : float, optional
        Percentile to compute (default: 95.0)
        
    Returns
    -------
    height_percentile : float
        Height percentile value in meters
        
    Examples
    --------
    >>> # 95th percentile height (common for canopy height)
    >>> h95 = compute_height_percentile(points, classification, percentile=95.0)
    >>> print(f"Canopy height: {h95:.2f}m")
    
    >>> # Median height
    >>> h50 = compute_height_percentile(points, classification, percentile=50.0)
    """
    height = compute_height_above_ground(points, classification)
    return np.percentile(height, percentile)


def compute_height_bins(
    points: np.ndarray,
    classification: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    num_bins: int = 10
) -> np.ndarray:
    """
    Compute height bin indices for each point.
    
    Useful for height-stratified analysis (e.g., vegetation layers).
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    classification : np.ndarray
        ASPRS classification codes
    bin_edges : np.ndarray, optional
        Custom bin edges. If None, creates evenly spaced bins.
    num_bins : int, optional
        Number of bins if bin_edges not provided (default: 10)
        
    Returns
    -------
    bin_indices : np.ndarray
        Bin index for each point, shape (N,)
        Values in range [0, num_bins-1]
        
    Examples
    --------
    >>> # 10 evenly spaced height bins
    >>> bins = compute_height_bins(points, classification, num_bins=10)
    
    >>> # Custom vegetation layer bins
    >>> layer_edges = np.array([0, 0.5, 2.0, 5.0, 15.0, 30.0])
    >>> bins = compute_height_bins(points, classification, bin_edges=layer_edges)
    >>> # bins[i] indicates vegetation layer: 0=ground, 1=understory, 2=mid, 3=canopy, 4=emergent
    """
    height = compute_height_above_ground(points, classification)
    
    if bin_edges is None:
        # Create evenly spaced bins
        max_height = np.max(height)
        bin_edges = np.linspace(0, max_height, num_bins + 1)
    
    # Assign points to bins
    bin_indices = np.digitize(height, bin_edges) - 1
    
    # Clamp to valid range
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    return bin_indices.astype(np.int32)
