"""
CPU-based feature computation strategy.

This strategy uses NumPy and scikit-learn for feature computation
on CPU. Best for small to medium datasets (< 1M points) or when
GPU is not available.

Author: IGN LiDAR HD Development Team
Date: October 21, 2025
Version: 3.1.0-dev (Week 2 refactoring)
"""

import logging
from typing import Dict, Optional

import numpy as np

from .compute.features import compute_all_features_optimized
from .compute.curvature import compute_curvature as compute_curvature_canonical
from .compute.eigenvalues import compute_eigenvalue_features
from .compute.height import compute_height_above_ground
from .compute.rgb_nir import compute_rgb_features
from .strategies import BaseFeatureStrategy
from ..utils.normalization import normalize_rgb

# Phase 3.4: Try to import vectorized CPU implementation
try:
    from .compute.vectorized_cpu import compute_all_features_vectorized
    VECTORIZED_CPU_AVAILABLE = True
except ImportError:
    VECTORIZED_CPU_AVAILABLE = False
    compute_all_features_vectorized = None

logger = logging.getLogger(__name__)


class CPUStrategy(BaseFeatureStrategy):
    """
    CPU-based feature computation using NumPy and scikit-learn.

    This strategy is optimal for:
    - Small datasets (< 1M points)
    - Systems without GPU
    - When GPU overhead is not justified

    Performance:
    - Small (< 100K points): 1-2 seconds
    - Medium (100K-1M points): 5-30 seconds
    - Large (> 1M points): 60+ seconds (use GPU instead)

    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        radius (float): Search radius for neighbor queries (0 = use k-NN)
        auto_k (bool): Automatically estimate optimal k based on density
        include_extra (bool): Compute extra features (slower but more complete)
        chunk_size (int): Process in chunks to reduce memory usage
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        auto_k: bool = True,
        include_extra: bool = False,
        chunk_size: int = 500_000,
        use_vectorized: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize CPU strategy.

        Args:
            k_neighbors: Number of neighbors for local features (if auto_k=False)
            radius: Search radius in meters (0 = use k-NN instead)
            auto_k: Automatically estimate optimal k based on point density
            include_extra: Compute expensive extra features
            chunk_size: Process in chunks for large datasets
            use_vectorized: Use Phase 3.4 vectorized CPU implementation (if available)
            verbose: Enable detailed logging
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)
        self.auto_k = auto_k
        self.include_extra = include_extra
        self.chunk_size = chunk_size
        self.use_vectorized = use_vectorized and VECTORIZED_CPU_AVAILABLE

        if verbose:
            logger.info(
                f"Initialized CPU strategy: k={k_neighbors}, radius={radius}m, "
                f"auto_k={auto_k}, chunk_size={chunk_size:,}, "
                f"vectorized={self.use_vectorized}"
            )

    def compute(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        classification: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using CPU.

        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            rgb: (N, 3) array of RGB values (optional)
            nir: (N,) array of near-infrared values (optional)
            classification: (N,) array of ASPRS classification codes (optional)
            **kwargs: Additional parameters

        Returns:
            Dictionary with feature arrays:
            - 'normals': (N, 3) surface normals
            - 'curvature': (N,) curvature values
            - 'height': (N,) height above ground
            - 'planarity': (N,) planarity scores
            - 'linearity': (N,) linearity scores
            - 'sphericity': (N,) sphericity scores
            - 'anisotropy': (N,) anisotropy scores
            - 'roughness': (N,) roughness values
            - 'density': (N,) local point density
            - 'verticality': (N,) verticality scores
            - 'horizontality': (N,) horizontality scores
            - Additional features if include_extra=True
        """
        n_points = len(points)

        if self.verbose:
            logger.info(
                f"Computing features for {n_points:,} points using CPU strategy"
            )

        # Determine k value (use default if auto_k not needed)
        k = self.k_neighbors

        # ✅ OPTIMIZATION (v3.5.3): Check for cached normals/eigenvalues
        cached_intermediates = self.get_cached_intermediates()
        
        if cached_intermediates is not None:
            # Use cached normals and eigenvalues - recompute only derived features
            normals, eigenvalues = cached_intermediates
            
            if self.verbose:
                logger.debug(f"♻️  Using cached normals/eigenvalues, computing only derived features")
            
            # ✅ Use canonical implementations (v3.5.3 consolidation)
            # Compute curvature from eigenvalues
            curvature = compute_curvature_canonical(eigenvalues, method='standard')
            
            # Compute all geometric features from eigenvalues using canonical implementation
            eigenvalue_features = compute_eigenvalue_features(
                eigenvalues, 
                epsilon=1e-10, 
                include_all=self.include_extra
            )
            
            # Build result from cached data + canonical features
            features = {
                "normals": normals,
                "eigenvalues": eigenvalues,
                "curvature": curvature,
                "planarity": eigenvalue_features["planarity"],
                "linearity": eigenvalue_features["linearity"],
                "sphericity": eigenvalue_features["sphericity"],
            }
            
            # Add additional features if requested (already computed by canonical implementation)
            if self.include_extra:
                features["anisotropy"] = eigenvalue_features["anisotropy"]
                features["omnivariance"] = eigenvalue_features["omnivariance"]
                features["eigenentropy"] = eigenvalue_features["eigenentropy"]
                features["verticality"] = np.clip(1.0 - np.abs(normals[:, 2]), 0.0, 1.0)
                features["roughness"] = curvature  # Alias
            
        else:
            # No cache - compute all features from scratch
            if self.verbose:
                logger.debug(f"Computing all features from scratch (k={k})")
            
            # Compute all geometric features using optimized function
            features = compute_all_features_optimized(
                points=points, k_neighbors=k, compute_advanced=self.include_extra
            )

        # Build result dictionary compatible with strategy interface
        # 🔧 FIX: Return ALL computed features, not just a subset
        result = {
            "normals": features["normals"].astype(np.float32),
            "curvature": features["curvature"].astype(np.float32),
            "planarity": features["planarity"].astype(np.float32),
            "linearity": features["linearity"].astype(np.float32),
            "sphericity": features["sphericity"].astype(np.float32),
        }

        # Add advanced features if requested
        if self.include_extra:
            result["anisotropy"] = features.get(
                "anisotropy", features["sphericity"]
            ).astype(np.float32)
            result["roughness"] = features.get(
                "roughness", features["curvature"]
            ).astype(np.float32)
            # ✅ FIXED: Verticality was inverted (walls got low scores)
            # Correct formula: verticality = 1 - |normal_z|
            result["verticality"] = features.get(
                "verticality",
                np.clip(1.0 - np.abs(features["normals"][:, 2]), 0.0, 1.0),
            ).astype(np.float32)
            result["density"] = features.get(
                "density", np.ones(n_points, dtype=np.float32)
            ).astype(np.float32)

            # 🆕 Add ALL other computed features (eigenvalues, normal components, etc.)
            for feat_name, feat_data in features.items():
                if (
                    feat_name not in result and feat_name != "normals"
                ):  # Avoid duplicating normals
                    result[feat_name] = feat_data.astype(np.float32)

        # Add height if we have classification
        height_method = kwargs.get("height_method", "ground_plane")
        if classification is not None and height_method == "dtm":
            # Optional pipeline parameter (see FeatureConfig.height_method):
            # per-point ground elevation from RGE ALTI/LiDAR HD MNT instead
            # of the single per-tile ground_z scalar below.
            result["height"] = compute_height_above_ground(
                points,
                classification,
                method="dtm",
                dtm_fetcher=kwargs.get("dtm_fetcher"),
                crs=kwargs.get("dtm_crs", "EPSG:2154"),
            )
        elif classification is not None:
            # Compute height above ground (unchanged default behavior)
            ground_mask = classification == 2
            if ground_mask.any():
                ground_z = points[ground_mask, 2].min()
                result["height"] = (points[:, 2] - ground_z).astype(np.float32)
            else:
                result["height"] = (points[:, 2] - points[:, 2].min()).astype(
                    np.float32
                )
        else:
            result["height"] = np.zeros(n_points, dtype=np.float32)

        # Compute RGB features if provided
        if rgb is not None:
            rgb_features = compute_rgb_features(rgb, use_gpu=False)
            result.update(rgb_features)

        # Compute NDVI if NIR and RGB provided
        if nir is not None and rgb is not None:
            red = rgb[:, 0]  # Assuming RGB order
            ndvi = self.compute_ndvi(nir, red)
            result["ndvi"] = ndvi.astype(np.float32)

        if self.verbose:
            logger.info(f"CPU computation complete: {len(result)} feature types")

        return result

    def compute_features(
        self,
        points: np.ndarray,
        classification: Optional[np.ndarray] = None,
        auto_k: bool = False,
        include_extra: bool = False,
        patch_center: Optional[np.ndarray] = None,
        mode: str = "lod2",
        radius: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Adapter method for compatibility with FeatureOrchestrator.

        This method provides the old interface expected by the orchestrator,
        while internally calling the new compute() method.

        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k
            include_extra: Whether to include extra features
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', etc.)
            radius: Search radius (overrides self.radius if provided)
            **kwargs: Additional parameters

        Returns:
            Dictionary with feature arrays
        """
        # Override settings if specified
        old_auto_k = self.auto_k
        old_include_extra = self.include_extra
        old_radius = self.radius

        if auto_k is not None:
            self.auto_k = auto_k
        if include_extra is not None:
            self.include_extra = include_extra
        if radius is not None:
            self.radius = radius

        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get("rgb", None)
        nir = kwargs.get("nir", None)
        intensities = kwargs.get("intensities", None)

        # Call the main compute method
        result = self.compute(
            points=points,
            intensities=intensities,
            rgb=rgb,
            nir=nir,
            classification=classification,
            **kwargs,
        )

        # Add distance_to_center if patch_center provided
        if patch_center is not None:
            distances = np.linalg.norm(points - patch_center, axis=1)
            result["distance_to_center"] = distances.astype(np.float32)

        # Restore original settings
        self.auto_k = old_auto_k
        self.include_extra = old_include_extra
        self.radius = old_radius

        return result


    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CPUStrategy(k_neighbors={self.k_neighbors}, radius={self.radius}, "
            f"auto_k={self.auto_k}, chunk_size={self.chunk_size:,})"
        )


# Export
__all__ = ["CPUStrategy"]
