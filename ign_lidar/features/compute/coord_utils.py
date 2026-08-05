"""
Coordinate recentering utilities for float32 geometry.

IGN LiDAR HD tiles are stored in Lambert-93 (EPSG:2154): X ≈ 6.5e5 m and
Y ≈ 6.9e6 m. A float32 carries ~7 significant decimal digits, so at that
magnitude two consecutive representable values are ~0.06 m apart in X and
~0.5 m apart in Y. Casting *absolute* coordinates to float32 therefore snaps
the cloud onto a coarse anisotropic grid, which bands the k-NN neighbourhoods
and stripes every PCA-derived feature (normals, curvature, planarity,
linearity, sphericity).

All of those features are translation-invariant, so the fix is always the
same: subtract a float64 origin *first*, then cast the (small) local
coordinates to float32. This module is the single source of truth for that
operation - it deliberately depends on numpy only, so it can be imported from
`features/` and from `optimization/` alike without circular imports.
"""

from typing import Optional, Tuple

import numpy as np

__all__ = ["recenter_to_local_f32"]


def recenter_to_local_f32(
    points: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract a float64 origin from `points` before casting them to float32.

    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (N, D). Never modified (a new array is returned).
    origin : np.ndarray, optional
        (D,) origin to subtract. If None, the centroid of `points` is used
        (accumulated in float64). Pass an explicit origin when several arrays
        must share the same local frame (e.g. reference points and query
        points of a k-NN search).

    Returns
    -------
    points_local : np.ndarray
        (N, D) float32 coordinates relative to `origin`.
    origin : np.ndarray
        (D,) float64 origin actually used. ``points_local + origin`` recovers
        the input coordinates, up to float32 *relative* precision (~1e-7 of
        the local extent, i.e. sub-millimetre for a 1 km tile) instead of the
        0.06-0.5 m quantisation of a direct cast.

    Notes
    -----
    Safe to apply defensively: recentering an already-local cloud is harmless
    (the origin is then close to zero), so callers do not need to know whether
    their input is absolute Lambert-93 or already local.
    """
    if origin is None:
        origin = points.mean(axis=0, dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64)

    # Subtract in float64 and cast on the fly: numpy buffers the cast
    # internally, so no full-size float64 temporary is materialised (matters
    # for 20M+ point tiles, where the prep pipeline is RAM-bound).
    points_local = np.empty(points.shape, dtype=np.float32)
    np.subtract(points, origin, out=points_local, casting="same_kind")

    return points_local, origin
