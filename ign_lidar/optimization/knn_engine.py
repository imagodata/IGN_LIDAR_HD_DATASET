"""
K-Nearest Neighbors Engine

Single source of truth for all KNN operations across the codebase.

This module consolidates 18+ scattered KNN/KDTree implementations into
a high-performance engine with:
- Multi-backend support (FAISS-GPU, cuML, sklearn)
- Automatic backend selection based on data size and hardware
- Consistent API across all backends
- Optimized memory management
- GPU acceleration when available

Replaces scattered implementations in:
- optimization/gpu_accelerated_ops.py (4 KNN functions)
- features/compute/faiss_knn.py (5 KNN functions)
- optimization/gpu_kdtree.py (GPUKDTree class)
- features/utils.py (build_kdtree, quick_kdtree)
- optimization/gpu_kernels.py (compute_knn_distances)
- io/formatters/*.py (2 files with KNN graph building)

Author: LiDAR Trainer Agent (Phase 2: KNN Consolidation)
Date: November 21, 2025
Version: 1.0
"""

import logging
from typing import Optional, Tuple, Literal
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Detection
# ============================================================================

class KNNBackend(Enum):
    """Available KNN backends."""
    FAISS_GPU = "faiss-gpu"      # Fastest (10-50x speedup)
    FAISS_CPU = "faiss-cpu"      # Fast (2-5x speedup)
    CUML = "cuml"                # GPU ML library
    SKLEARN = "sklearn"          # CPU fallback (always available)
    AUTO = "auto"                # Automatic selection


# Check backend availability
HAS_FAISS = False
HAS_FAISS_GPU = False
HAS_CUML = False

try:
    import faiss
    HAS_FAISS = True
    HAS_FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    pass

try:
    from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors
    HAS_CUML = True
except ImportError:
    pass

# sklearn is always available (required dependency)
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# KNN Engine
# ============================================================================

class KNNEngine:
    """
    K-Nearest Neighbors engine with multi-backend support.
    
    This class provides a single, consistent API for KNN operations
    across all backends (FAISS-GPU, FAISS-CPU, cuML, sklearn).
    
    Features:
    - Automatic backend selection based on data size and hardware
    - Consistent API regardless of backend
    - Optimized memory management
    - GPU acceleration when available
    - Graceful fallback to CPU
    
    Example:
        >>> from ign_lidar.optimization import KNNEngine
        >>> 
        >>> # Automatic backend selection
        >>> engine = KNNEngine()
        >>> distances, indices = engine.search(points, k=30)
        >>> 
        >>> # Force specific backend
        >>> engine = KNNEngine(backend='faiss-gpu')
        >>> distances, indices = engine.search(points, k=30)
    """
    
    def __init__(
        self,
        backend: Optional[Literal['auto', 'faiss-gpu', 'faiss-cpu', 'cuml', 'sklearn']] = 'auto',
        metric: str = 'euclidean',
        n_jobs: int = -1
    ):
        """
        Initialize KNN engine.
        
        Args:
            backend: Backend to use ('auto' for automatic selection)
            metric: Distance metric ('euclidean' or 'cosine')
            n_jobs: Number of CPU threads (-1 = all cores)
        """
        self.backend_preference = backend
        self.metric = metric
        self.n_jobs = n_jobs
        self._index = None
        self._fitted = False
        self._points = None
        
        logger.info(f"KNNEngine initialized (backend={backend}, metric={metric})")
    
    def _select_backend(
        self,
        n_points: int,
        n_dims: int,
        k: int
    ) -> KNNBackend:
        """
        Select optimal backend based on data characteristics.
        
        Selection criteria:
        - FAISS-GPU: Large datasets (>100k) with GPU available
        - cuML: Medium datasets (10k-100k) with GPU available
        - FAISS-CPU: Large datasets (>50k) without GPU
        - sklearn: Small datasets (<50k) or fallback
        
        Args:
            n_points: Number of points
            n_dims: Number of dimensions
            k: Number of neighbors
            
        Returns:
            Selected backend
        """
        if self.backend_preference != 'auto':
            # User specified backend
            backend_map = {
                'faiss-gpu': KNNBackend.FAISS_GPU,
                'faiss-cpu': KNNBackend.FAISS_CPU,
                'cuml': KNNBackend.CUML,
                'sklearn': KNNBackend.SKLEARN
            }
            return backend_map.get(self.backend_preference, KNNBackend.SKLEARN)
        
        # Automatic selection based on data size and hardware
        if n_points >= 100_000:
            # Large dataset - prefer GPU
            if HAS_FAISS_GPU and self.metric == 'euclidean':
                return KNNBackend.FAISS_GPU
            elif HAS_CUML and self.metric == 'euclidean':
                return KNNBackend.CUML
            elif HAS_FAISS:
                return KNNBackend.FAISS_CPU
        elif n_points >= 50_000:
            # Medium dataset
            if HAS_FAISS:
                return KNNBackend.FAISS_CPU
        
        # Small dataset or fallback
        return KNNBackend.SKLEARN
    
    def fit(self, points: np.ndarray) -> 'KNNEngine':
        """
        Fit the KNN index on reference points.
        
        Args:
            points: Reference points [N, D]
            
        Returns:
            self (for chaining)
            
        Example:
            >>> engine = KNNEngine()
            >>> engine.fit(reference_points)
            >>> distances, indices = engine.query(query_points, k=30)
        """
        if points.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {points.shape}")
        
        self._points = points
        self._fitted = False
        self._index = None
        
        logger.debug(f"KNN index fitted on {len(points)} points with {points.shape[1]} dimensions")
        return self
    
    def search(
        self,
        points: np.ndarray,
        k: int = 30,
        query_points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k-nearest neighbors.
        
        This is the main API for KNN search. It automatically:
        1. Selects optimal backend
        2. Builds/reuses index
        3. Performs search
        4. Returns results in consistent format
        
        Args:
            points: Reference points [N, D] (if not fitted) or query points (if fitted)
            k: Number of nearest neighbors
            query_points: Query points [M, D] (optional, for fit-then-query workflow)
            
        Returns:
            Tuple of:
            - distances: [M, k] distances to neighbors
            - indices: [M, k] indices of neighbors
            
        Example:
            >>> engine = KNNEngine()
            >>> 
            >>> # Self-query (most common)
            >>> distances, indices = engine.search(points, k=30)
            >>> 
            >>> # Separate query set
            >>> engine.fit(reference_points)
            >>> distances, indices = engine.search(query_points, k=30)
        """
        # Handle fit-then-query workflow
        if query_points is None:
            if self._fitted:
                # Already fitted, use points as query
                query_points = points
                points = self._points
            else:
                # Self-query
                query_points = points
        
        if points.ndim != 2 or query_points.ndim != 2:
            raise ValueError("points and query_points must be 2D arrays")
        
        n_points, n_dims = points.shape
        n_queries = len(query_points)
        
        if k >= n_points:
            raise ValueError(f"k={k} must be less than n_points={n_points}")
        
        # Select backend
        backend = self._select_backend(n_points, n_dims, k)
        
        logger.debug(
            f"KNN search: {n_queries} queries, {n_points} points, "
            f"k={k}, backend={backend.value}"
        )
        
        # Dispatch to appropriate backend
        if backend == KNNBackend.FAISS_GPU:
            return self._search_faiss_gpu(points, query_points, k)
        elif backend == KNNBackend.FAISS_CPU:
            return self._search_faiss_cpu(points, query_points, k)
        elif backend == KNNBackend.CUML:
            return self._search_cuml(points, query_points, k)
        else:  # SKLEARN
            return self._search_sklearn(points, query_points, k)
    
    def radius_search(
        self,
        points: np.ndarray,
        radius: float,
        query_points: Optional[np.ndarray] = None,
        max_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for all neighbors within a given radius.
        
        Args:
            points: Reference points [N, D] (if not fitted) or query points (if fitted)
            radius: Search radius (same units as point coordinates)
            query_points: Query points [M, D] (optional)
            max_neighbors: Maximum neighbors per query (None = unlimited)
            
        Returns:
            Tuple of:
            - distances: List/array of distances (variable length per query)
            - indices: List/array of indices (variable length per query)
            
        Note:
            Returns may be list of arrays (sklearn/cuML) or padded arrays
            (FAISS with -1 for invalid entries).
            
        Example:
            >>> engine = KNNEngine()
            >>> 
            >>> # Search within 3 meters
            >>> distances, indices = engine.radius_search(points, radius=3.0)
            >>> 
            >>> # Limit to 50 neighbors max
            >>> distances, indices = engine.radius_search(
            ...     points, radius=5.0, max_neighbors=50
            ... )
        """
        # Handle fit-then-query workflow
        if query_points is None:
            if self._fitted:
                query_points = points
                points = self._points
            else:
                query_points = points
        
        if points.ndim != 2 or query_points.ndim != 2:
            raise ValueError("points and query_points must be 2D arrays")
        
        n_points, n_dims = points.shape
        n_queries = len(query_points)
        
        # Select backend
        # Note: cuML NearestNeighbors doesn't have radius_neighbors method,
        # so we always use sklearn for radius search
        backend = KNNBackend.SKLEARN
        
        logger.debug(
            f"Radius search: {n_queries} queries, {n_points} points, "
            f"radius={radius}, backend={backend.value}"
        )
        
        # Always use sklearn for radius search
        return self._radius_search_sklearn(points, query_points, radius, max_neighbors)
    
    def _radius_search_cuml(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        radius: float,
        max_neighbors: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Radius search using cuML (GPU)."""
        from cuml.neighbors import NearestNeighbors as cuMLNN
        
        logger.debug(f"Using cuML radius search for {len(points)} points")
        
        nn = cuMLNN(
            algorithm='brute',
            metric=self.metric
        )
        nn.fit(points)
        distances, indices = nn.radius_neighbors(
            query_points,
            radius=radius,
            return_distance=True
        )
        
        # Convert to numpy
        if hasattr(distances, 'get'):
            distances = [d.get() for d in distances]
            indices = [i.get() for i in indices]
        
        # Apply max_neighbors limit if specified
        if max_neighbors is not None:
            distances = [d[:max_neighbors] for d in distances]
            indices = [i[:max_neighbors] for i in indices]
        
        return distances, indices
    
    def _radius_search_sklearn(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        radius: float,
        max_neighbors: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Radius search using sklearn (CPU fallback)."""
        logger.debug(f"Using sklearn radius search for {len(points)} points")
        
        nn = NearestNeighbors(
            algorithm='auto',
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        nn.fit(points)
        distances, indices = nn.radius_neighbors(
            query_points,
            radius=radius,
            return_distance=True
        )
        
        # Apply max_neighbors limit if specified
        if max_neighbors is not None:
            distances = [d[:max_neighbors] for d in distances]
            indices = [i[:max_neighbors] for i in indices]
        
        return distances, indices
    
    def _to_float32_for_faiss(
        self,
        points: np.ndarray,
        query_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cast reference and query points to the float32 FAISS requires.

        Absolute Lambert-93 coordinates (X ≈ 6.5e5, Y ≈ 6.9e6) quantise to
        0.06-0.5 m once cast to float32, which bands the returned
        neighbourhoods. L2 neighbourhoods (and squared L2 distances) are
        translation-invariant, so both arrays are moved to a common local
        frame first — same origin for both, otherwise the query would be
        matched against a shifted index. Inner-product search is NOT
        translation-invariant and keeps the raw cast.
        """
        if self.metric != 'euclidean':
            return points.astype(np.float32), query_points.astype(np.float32)

        from ign_lidar.features.compute.coord_utils import recenter_to_local_f32

        points_f32, origin = recenter_to_local_f32(points)
        query_f32, _ = recenter_to_local_f32(query_points, origin=origin)
        return points_f32, query_f32

    def _search_faiss_gpu(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS-GPU (fastest)."""
        from ign_lidar.optimization.faiss_utils import create_faiss_index

        logger.debug(f"Using FAISS-GPU backend for {len(points)} points")

        # Create optimized index
        index, res = create_faiss_index(
            n_dims=points.shape[1],
            n_points=len(points),
            use_gpu=True,
            approximate=len(points) > 100_000,
            metric='L2' if self.metric == 'euclidean' else 'IP'
        )

        # Convert to float32 (FAISS requirement)
        points_f32, query_f32 = self._to_float32_for_faiss(points, query_points)

        # Train if needed (IVF index)
        if hasattr(index, 'is_trained') and not index.is_trained:
            index.train(points_f32)
        
        # Add data
        index.add(points_f32)
        
        # Search
        distances, indices = index.search(query_f32, k)
        
        return distances, indices
    
    def _search_faiss_cpu(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS-CPU (fast)."""
        import faiss

        logger.debug(f"Using FAISS-CPU backend for {len(points)} points")

        points_f32, query_f32 = self._to_float32_for_faiss(points, query_points)

        # Create flat index for exact search
        if self.metric == 'euclidean':
            index = faiss.IndexFlatL2(points.shape[1])
        else:
            index = faiss.IndexFlatIP(points.shape[1])
        
        index.add(points_f32)
        distances, indices = index.search(query_f32, k)
        
        return distances, indices
    
    def _search_cuml(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using cuML (GPU)."""
        from cuml.neighbors import NearestNeighbors as cuMLNN

        logger.debug(f"Using cuML backend for {len(points)} points")

        # Same metric-gated recentering as the FAISS backends (see
        # _to_float32_for_faiss): cuML casts to float32 internally, and only
        # euclidean/L2 neighborhoods are translation-invariant.
        points_r, query_r = self._to_float32_for_faiss(points, query_points)

        nn = cuMLNN(n_neighbors=k, algorithm='brute', metric=self.metric)
        nn.fit(points_r)
        distances, indices = nn.kneighbors(query_r)

        # Convert to numpy if needed (cuML returns cupy/cudf arrays/frames).
        if hasattr(distances, 'get'):
            distances = distances.get()
        if hasattr(indices, 'get'):
            indices = indices.get()

        return distances, indices
    
    def _search_sklearn(
        self,
        points: np.ndarray,
        query_points: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using sklearn (CPU fallback)."""
        logger.debug(f"Using sklearn backend for {len(points)} points")
        
        nn = NearestNeighbors(
            n_neighbors=k,
            algorithm='auto',
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        nn.fit(points)
        distances, indices = nn.kneighbors(query_points)
        
        return distances, indices


# ============================================================================
# Convenience Functions
# ============================================================================

def knn_search(
    points: np.ndarray,
    k: int = 30,
    query_points: Optional[np.ndarray] = None,
    backend: str = 'auto',
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for one-off KNN searches.
    
    For repeated searches on the same dataset, use KNNEngine directly
    to avoid rebuilding the index.
    
    Args:
        points: Reference points [N, D]
        k: Number of nearest neighbors
        query_points: Query points [M, D] (None = self-query)
        backend: Backend to use ('auto', 'faiss-gpu', 'cuml', 'sklearn')
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Tuple of (distances, indices)
        
    Example:
        >>> from ign_lidar.optimization import knn_search
        >>> 
        >>> # Self-query (most common)
        >>> distances, indices = knn_search(points, k=30)
        >>> 
        >>> # Separate query set
        >>> distances, indices = knn_search(
        ...     points=reference_points,
        ...     query_points=query_points,
        ...     k=30
        ... )
        >>> 
        >>> # Force GPU
        >>> distances, indices = knn_search(points, k=30, backend='faiss-gpu')
    """
    engine = KNNEngine(backend=backend, metric=metric)
    return engine.search(points, k=k, query_points=query_points)


def build_knn_graph(
    points: np.ndarray,
    k: int = 30,
    backend: str = 'auto'
) -> np.ndarray:
    """
    Build k-nearest neighbors graph.
    
    This is commonly used for graph-based neural networks and
    datasets that require connectivity information.
    
    Args:
        points: Points [N, D]
        k: Number of neighbors per point
        backend: Backend to use ('auto', 'faiss-gpu', 'cuml', 'sklearn')
        
    Returns:
        Neighbor indices [N, k]
        
    Example:
        >>> from ign_lidar.optimization import build_knn_graph
        >>> 
        >>> # Build graph
        >>> neighbors = build_knn_graph(points, k=30)
        >>> 
        >>> # Use in neural network
        >>> edge_index = create_edge_index(neighbors)
    """
    _, indices = knn_search(points, k=k, backend=backend)
    return indices


def radius_search(
    points: np.ndarray,
    radius: float,
    query_points: Optional[np.ndarray] = None,
    backend: str = 'auto',
    metric: str = 'euclidean',
    max_neighbors: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for radius-based neighbor search.
    
    Finds all neighbors within a given radius for each query point.
    
    Args:
        points: Reference points [N, D]
        radius: Search radius (same units as coordinates)
        query_points: Query points [M, D] (None = self-query)
        backend: Backend to use ('auto', 'cuml', 'sklearn')
        metric: Distance metric ('euclidean' or 'cosine')
        max_neighbors: Maximum neighbors per query (None = unlimited)
        
    Returns:
        Tuple of (distances, indices) - lists of variable-length arrays
        
    Example:
        >>> from ign_lidar.optimization import radius_search
        >>> 
        >>> # Find all neighbors within 3 meters
        >>> distances, indices = radius_search(points, radius=3.0)
        >>> 
        >>> # Limit to 50 neighbors per point
        >>> distances, indices = radius_search(
        ...     points, radius=5.0, max_neighbors=50
        ... )
        >>> 
        >>> # Access results for first query point
        >>> print(f"Query 0 has {len(indices[0])} neighbors")
        >>> print(f"Closest neighbor at distance {distances[0][0]:.2f}m")
    """
    engine = KNNEngine(backend=backend, metric=metric)
    return engine.radius_search(
        points,
        radius=radius,
        query_points=query_points,
        max_neighbors=max_neighbors
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'KNNEngine',
    'KNNBackend',
    'knn_search',
    'radius_search',
    'build_knn_graph',
    'HAS_FAISS',
    'HAS_FAISS_GPU',
    'HAS_CUML',
]
