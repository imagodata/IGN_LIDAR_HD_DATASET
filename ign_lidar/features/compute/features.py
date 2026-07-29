"""
Optimized feature computation using Numba JIT compilation.

This module provides ultra-fast computation of geometric features using
JIT compilation and parallel execution. It combines normal vectors,
eigenvalues, curvature, and shape features in optimized implementations.

Performance gains over standard implementations:
- Normals: 3-5x faster
- Complete features: 5-8x faster (single-pass computation)
- All features derived from shared covariance/eigenvalue computation

This is the RECOMMENDED way to compute features.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)

try:
    from numba import config, jit, prange

    NUMBA_AVAILABLE = True
    config.THREADING_LAYER = "threadsafe"

    def _ensure_working_threading_layer() -> None:
        """`threadsafe` (tbb/omp) is preferred so numba-jitted parallel code
        stays safe to call from multiple OS threads in the same process
        (e.g. the orchestrator's RGB/NIR `ThreadPoolExecutor` prefetching
        running alongside feature computation). Not every environment ships
        the tbb/openmp system libraries though (observed: a bare Ubuntu
        26.04 cloud image with no conda) — numba then raises `ValueError`
        on the *first* parallel dispatch instead of degrading gracefully,
        which previously broke every single call, not just the unsafe
        concurrent-thread case. Probe once at import time (before any real
        work) and fall back to `workqueue` — not thread-safe for concurrent
        multi-threaded numba calls, but correct for the common case of a
        single caller per process (e.g. one worker in a
        `multiprocessing.Pool`).
        """
        probe_input = np.zeros(4, dtype=np.float64)

        @jit(nopython=True, parallel=True)
        def _probe(x):
            out = np.zeros_like(x)
            for i in prange(len(x)):
                out[i] = x[i]
            return out

        try:
            _probe(probe_input)
        except ValueError:
            logger.warning(
                "Numba 'threadsafe' threading layer (tbb/omp) unavailable in "
                "this environment - falling back to 'workqueue'. Parallel "
                "feature computation still works; calling numba-jitted code "
                "concurrently from multiple threads in the same process is "
                "no longer guaranteed safe."
            )
            config.THREADING_LAYER = "workqueue"
            _probe(probe_input)  # raises again if even workqueue is broken

    _ensure_working_threading_layer()
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - using slow fallback implementation")


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_normals_and_eigenvalues_jit(
    points: np.ndarray, neighbor_indices: np.ndarray, k_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled normal and eigenvalue computation.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    neighbor_indices : np.ndarray
        Neighbor indices from KNN search, shape (N, k)
    k_neighbors : int
        Number of neighbors (k)

    Returns
    -------
    normals : np.ndarray
        Unit normal vectors, shape (N, 3)
    eigenvalues : np.ndarray
        Eigenvalues sorted descending, shape (N, 3)
    """
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float32)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)

    for i in prange(n_points):
        neighbors = points[neighbor_indices[i]]

        # Compute centroid
        centroid = np.zeros(3, dtype=np.float32)
        for j in range(k_neighbors):
            centroid[0] += neighbors[j, 0]
            centroid[1] += neighbors[j, 1]
            centroid[2] += neighbors[j, 2]
        centroid /= k_neighbors

        # Center neighbors
        centered = np.zeros((k_neighbors, 3), dtype=np.float32)
        for j in range(k_neighbors):
            centered[j, 0] = neighbors[j, 0] - centroid[0]
            centered[j, 1] = neighbors[j, 1] - centroid[1]
            centered[j, 2] = neighbors[j, 2] - centroid[2]

        # Compute covariance matrix
        cov = np.zeros((3, 3), dtype=np.float32)
        for j in range(k_neighbors):
            for a in range(3):
                for b in range(3):
                    cov[a, b] += centered[j, a] * centered[j, b]
        cov /= k_neighbors

        # Eigendecomposition (returns ascending order)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Store eigenvalues in DESCENDING order (λ1 >= λ2 >= λ3)
        eigenvalues[i, 0] = eigvals[2]  # Largest
        eigenvalues[i, 1] = eigvals[1]  # Middle
        eigenvalues[i, 2] = eigvals[0]  # Smallest

        # Normal is eigenvector of smallest eigenvalue
        normals[i, 0] = eigvecs[0, 0]
        normals[i, 1] = eigvecs[1, 0]
        normals[i, 2] = eigvecs[2, 0]

        # Normalize normal
        norm = np.sqrt(normals[i, 0] ** 2 + normals[i, 1] ** 2 + normals[i, 2] ** 2)
        if norm > 0:
            normals[i, 0] /= norm
            normals[i, 1] /= norm
            normals[i, 2] /= norm
        else:
            normals[i, 2] = 1.0

    return normals, eigenvalues


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_all_features_jit(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    k_neighbors: int,
    epsilon: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled feature computation.

    Computes normals, eigenvalues, curvature, planarity, linearity, and
    sphericity all in one pass from shared covariance computation.

    Parameters
    ----------
    points : np.ndarray
        Point cloud, shape (N, 3) - FULL point cloud for indexing
    neighbor_indices : np.ndarray
        Neighbor indices from KNN, shape (M, k) where M is the number of
        query points (may be a chunk of the full point cloud)
    k_neighbors : int
        Number of neighbors
    epsilon : float
        Small value to prevent division by zero

    Returns
    -------
    normals : np.ndarray (M, 3) - same size as neighbor_indices
    eigenvalues : np.ndarray (M, 3) - sorted descending
    curvature : np.ndarray (M,)
    planarity : np.ndarray (M,)
    linearity : np.ndarray (M,)
    sphericity : np.ndarray (M,)

    Note:
        The output arrays are sized based on neighbor_indices (M points),
        NOT the full points array (N points). This allows chunked processing.
    """
    # CRITICAL FIX: Use neighbor_indices size, NOT points size!
    # This allows chunked processing where we query a subset of points
    # but index into the full points array using the returned indices.
    n_query_points = neighbor_indices.shape[0]

    # Output arrays sized for the queried points (chunk size)
    normals = np.zeros((n_query_points, 3), dtype=np.float32)
    eigenvalues = np.zeros((n_query_points, 3), dtype=np.float32)
    curvature = np.zeros(n_query_points, dtype=np.float32)
    planarity = np.zeros(n_query_points, dtype=np.float32)
    linearity = np.zeros(n_query_points, dtype=np.float32)
    sphericity = np.zeros(n_query_points, dtype=np.float32)

    # Process queried points in parallel
    for i in prange(n_query_points):
        neighbors = points[neighbor_indices[i]]

        # Compute centroid
        centroid = np.zeros(3, dtype=np.float32)
        for j in range(k_neighbors):
            centroid[0] += neighbors[j, 0]
            centroid[1] += neighbors[j, 1]
            centroid[2] += neighbors[j, 2]
        centroid /= k_neighbors

        # Center neighbors
        centered = np.zeros((k_neighbors, 3), dtype=np.float32)
        for j in range(k_neighbors):
            centered[j, 0] = neighbors[j, 0] - centroid[0]
            centered[j, 1] = neighbors[j, 1] - centroid[1]
            centered[j, 2] = neighbors[j, 2] - centroid[2]

        # Compute covariance matrix
        cov = np.zeros((3, 3), dtype=np.float32)
        for j in range(k_neighbors):
            for a in range(3):
                for b in range(3):
                    cov[a, b] += centered[j, a] * centered[j, b]
        cov /= k_neighbors

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Store eigenvalues in DESCENDING order (λ1 >= λ2 >= λ3)
        lambda1 = eigvals[2]
        lambda2 = eigvals[1]
        lambda3 = eigvals[0]

        eigenvalues[i, 0] = lambda1
        eigenvalues[i, 1] = lambda2
        eigenvalues[i, 2] = lambda3

        # Normal is eigenvector of smallest eigenvalue
        normals[i, 0] = eigvecs[0, 0]
        normals[i, 1] = eigvecs[1, 0]
        normals[i, 2] = eigvecs[2, 0]

        # Normalize normal
        norm = np.sqrt(normals[i, 0] ** 2 + normals[i, 1] ** 2 + normals[i, 2] ** 2)
        if norm > 0:
            normals[i, 0] /= norm
            normals[i, 1] /= norm
            normals[i, 2] /= norm
        else:
            normals[i, 2] = 1.0

        # Compute features from eigenvalues
        sum_eig = lambda1 + lambda2 + lambda3 + epsilon

        # Curvature: λ3 / (λ1 + λ2 + λ3)
        curvature[i] = lambda3 / sum_eig

        # Planarity: (λ2 - λ3) / λ1
        if lambda1 > epsilon:
            planarity[i] = (lambda2 - lambda3) / lambda1

        # Linearity: (λ1 - λ2) / λ1
        if lambda1 > epsilon:
            linearity[i] = (lambda1 - lambda2) / lambda1

        # Sphericity: λ3 / λ1
        if lambda1 > epsilon:
            sphericity[i] = lambda3 / lambda1

    return normals, eigenvalues, curvature, planarity, linearity, sphericity


# ============================================================================
# DEPRECATED: compute_normals() removed from this module (Phase 2 consolidation)
# ============================================================================
# 
# This function was a PURE DUPLICATION of features/compute/normals.py
# It has been removed as part of the Phase 2 code consolidation (Nov 2025).
#
# USE INSTEAD:
#   from ign_lidar.features.compute import compute_normals  # Routes to normals.py
#   from ign_lidar.features.compute.normals import compute_normals  # Direct import
#
# The compute_normals() function is now imported from normals.py in __init__.py
# ============================================================================


def compute_all_features_optimized(
    points: np.ndarray,
    k_neighbors: int = 20,
    compute_advanced: bool = True,
    chunk_size: int = 2_000_000,  # Optimized for 64GB systems
) -> Dict[str, np.ndarray]:
    """
    ✅ **CANONICAL CPU IMPLEMENTATION (v3.5.2+)** - JIT-compiled feature computation.
    
    This is the LOW-LEVEL OPTIMIZED implementation for CPU-based feature computation.
    DO NOT DUPLICATE - this is the single source of truth for CPU feature computation.
    
    For high-level API with automatic mode selection (CPU/GPU/boundary-aware),
    use dispatcher.compute_all_features() instead.

    Relationship:
    - dispatcher.compute_all_features() → HIGH-LEVEL (mode selection dispatcher)
    - features.compute_all_features_optimized() → LOW-LEVEL (CPU implementation) ✅ THIS FILE
    - gpu_kernels.compute_normals_eigenvalues_fused() → LOW-LEVEL (GPU implementation)
    - These are COMPLEMENTARY, not duplicates!

    This is 5-8x faster than calling individual feature functions because:
    1. KD-tree built only once
    2. Neighbor search done only once (chunked for large datasets)
    3. Covariance/eigenvalues computed only once
    4. All features derived from shared eigenvalues
    5. JIT compilation with parallel execution

    MEMORY OPTIMIZED: For large point clouds (>2M points), processes in chunks
    to avoid memory overflow. KD-tree is built once, but neighbor queries
    are batched for efficiency.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    k_neighbors : int
        Number of nearest neighbors (default: 20)
    compute_advanced : bool
        Whether to compute advanced features (anisotropy, roughness, etc.)
    chunk_size : int
        Points per chunk for neighbor queries (default: 2M for 64GB systems)
        Adjust based on available RAM

    Returns
    -------
    features : dict
        Dictionary containing:
        - 'normals': (N, 3) normal vectors
        - 'normal_x', 'normal_y', 'normal_z': (N,) normal components
        - 'eigenvalues': (N, 3) eigenvalues (descending)
        - 'curvature': (N,) surface curvature
        - 'planarity': (N,) planarity
        - 'linearity': (N,) linearity
        - 'sphericity': (N,) sphericity
        - 'anisotropy': (N,) anisotropy (if compute_advanced=True)
        - 'roughness': (N,) roughness (if compute_advanced=True)
        - 'verticality': (N,) verticality (if compute_advanced=True)

    Examples
    --------
    >>> points = np.random.rand(100000, 3).astype(np.float32)
    >>> features = compute_all_features(points, k_neighbors=20)
    >>> assert 'normals' in features
    >>> assert 'curvature' in features
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba is required for optimized features. "
            "Install with: conda install -c conda-forge numba"
        )

    # Input validation
    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.shape[0] < k_neighbors:
        raise ValueError(
            f"Not enough points ({points.shape[0]}) for k_neighbors={k_neighbors}"
        )

    n_points = points.shape[0]

    # Ensure float32 for performance
    points = points.astype(np.float32, copy=False)

    # Build KD-tree ONCE (fast and memory-efficient)
    from sklearn.neighbors import NearestNeighbors
    import multiprocessing

    # ✅ FIXED: Avoid sklearn parallelism conflicts with multiprocessing
    # If we're in a worker process, use n_jobs=1, otherwise use all cores
    current_process = multiprocessing.current_process()
    n_jobs = 1 if current_process.name != "MainProcess" else -1

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree", n_jobs=n_jobs)
    nbrs.fit(points)

    # MEMORY SAFETY: Query neighbors in chunks for large point clouds
    if n_points > chunk_size:
        logger.debug(
            f"  Processing {n_points:,} points in chunks of {chunk_size:,} for memory safety"
        )

        # Pre-allocate output arrays
        normals = np.zeros((n_points, 3), dtype=np.float32)
        eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
        curvature = np.zeros(n_points, dtype=np.float32)
        planarity = np.zeros(n_points, dtype=np.float32)
        linearity = np.zeros(n_points, dtype=np.float32)
        sphericity = np.zeros(n_points, dtype=np.float32)

        # Store median distances for density computation (if compute_advanced=True)
        all_median_dists = None
        if compute_advanced:
            all_median_dists = np.zeros(n_points, dtype=np.float32)

        # Process in chunks
        for start_idx in range(0, n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, n_points)
            chunk_points = points[start_idx:end_idx]

            # Query neighbors for chunk
            distances, indices = nbrs.kneighbors(chunk_points)

            # Store median distances for this chunk (before computing features)
            if compute_advanced:
                all_median_dists[start_idx:end_idx] = np.median(distances, axis=1)

            # CRITICAL FIX: Pass full 'points' array to JIT function because 'indices'
            # contains global indices into the full array, NOT local chunk indices.
            # The JIT function uses: points[neighbor_indices[i]] which requires the full array.
            (
                chunk_normals,
                chunk_eigenvalues,
                chunk_curvature,
                chunk_planarity,
                chunk_linearity,
                chunk_sphericity,
            ) = _compute_all_features_jit(
                points, indices, k_neighbors  # Correctly pass full points array
            )

            # Store chunk results
            normals[start_idx:end_idx] = chunk_normals
            eigenvalues[start_idx:end_idx] = chunk_eigenvalues
            curvature[start_idx:end_idx] = chunk_curvature
            planarity[start_idx:end_idx] = chunk_planarity
            linearity[start_idx:end_idx] = chunk_linearity
            sphericity[start_idx:end_idx] = chunk_sphericity

            # Explicit cleanup
            del distances, indices, chunk_normals, chunk_eigenvalues
            del chunk_curvature, chunk_planarity, chunk_linearity, chunk_sphericity
    else:
        # Small point cloud: process all at once (original fast path)
        distances, indices = nbrs.kneighbors(points)

        # Store median distances for density computation
        all_median_dists = np.median(distances, axis=1) if compute_advanced else None

        normals, eigenvalues, curvature, planarity, linearity, sphericity = (
            _compute_all_features_jit(points, indices, k_neighbors)
        )
        del distances, indices

    # Build feature dictionary
    features = {
        "normals": normals,
        "normal_x": normals[:, 0],
        "normal_y": normals[:, 1],
        "normal_z": normals[:, 2],
        "eigenvalues": eigenvalues,
        "eigenvalue_1": eigenvalues[:, 0],
        "eigenvalue_2": eigenvalues[:, 1],
        "eigenvalue_3": eigenvalues[:, 2],
        "curvature": curvature,
        "planarity": planarity,
        "linearity": linearity,
        "sphericity": sphericity,
    }

    # Compute advanced features if requested
    if compute_advanced:
        # Anisotropy: (λ1 - λ3) / λ1
        lambda1 = eigenvalues[:, 0]
        lambda3 = eigenvalues[:, 2]
        anisotropy = np.zeros_like(lambda1)
        mask = lambda1 > 1e-10
        anisotropy[mask] = (lambda1[mask] - lambda3[mask]) / lambda1[mask]
        features["anisotropy"] = anisotropy

        # Roughness: λ3 / (λ1 + λ2 + λ3) (same as curvature)
        features["roughness"] = curvature

        # Verticality: 1 - |normal_z|
        features["verticality"] = 1.0 - np.abs(normals[:, 2])

        # Density (local point density from neighbor distances)
        # Use pre-computed median distances from chunked/non-chunked processing
        if all_median_dists is not None:
            features["density"] = 1.0 / (all_median_dists**3 + 1e-10)
        else:
            # Fallback: estimate from eigenvalues (less accurate but functional)
            features["density"] = 1.0 / (eigenvalues[:, 0] + 1e-10)

    return features


def benchmark_features(
    points: np.ndarray, k_neighbors: int = 20, n_runs: int = 3
) -> dict:
    """
    Benchmark optimized feature computation.

    Parameters
    ----------
    points : np.ndarray
        Test point cloud
    k_neighbors : int
        Number of neighbors
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    results : dict
        Timing comparison and speedup
    """
    import time

    print(f"\n🔬 Benchmarking feature computation on {len(points):,} points...")
    print(f"   k_neighbors = {k_neighbors}")
    print(f"   n_runs = {n_runs}\n")

    # Warm up JIT
    print("⏳ Warming up JIT compiler...")
    sample = points[:1000].copy()
    _ = compute_all_features(sample, k_neighbors=min(k_neighbors, 100))
    print("✅ JIT warmup complete\n")

    # Benchmark
    print("📊 Benchmarking feature computation...")
    times = []
    for run in range(n_runs):
        start = time.perf_counter()
        features = compute_all_features(points, k_neighbors=k_neighbors)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        throughput = len(points) / elapsed
        print(f"   Run {run+1}/{n_runs}: {elapsed:.3f}s ({throughput:,.0f} pts/sec)")

    avg_time = np.mean(times)
    throughput = len(points) / avg_time
    print(f"   Average: {avg_time:.3f}s ({throughput:,.0f} pts/sec)\n")

    print("=" * 70)
    print(f"🎯 RESULTS:")
    print(f"   Throughput: {throughput:>10,.0f} pts/sec")
    print(f"   Time:       {avg_time:>10.3f} seconds")
    print("=" * 70)

    return {"throughput": throughput, "time": avg_time, "n_features": len(features)}


if __name__ == "__main__":
    print("🧪 Testing optimized feature computation...\n")

    # Generate test data
    n_points = 100000
    print(f"Generating {n_points:,} test points...")
    np.random.seed(42)
    points = np.random.rand(n_points, 3).astype(np.float32) * 10.0

    # Benchmark
    results = benchmark_features(points, k_neighbors=20, n_runs=3)

    print(f"\n✅ Computed {results['n_features']} feature types successfully!")
