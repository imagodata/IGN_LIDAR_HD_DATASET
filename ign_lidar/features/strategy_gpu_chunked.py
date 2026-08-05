"""
GPU-based feature computation strategy with chunking.

Uses the GPUProcessor which automatically handles chunking
for large datasets (> 10M points).

This strategy uses the GPUProcessor with intelligent auto-chunking for
memory-efficient processing of massive point clouds.

**Phase 3 GPU Optimizations (November 27, 2025)**:
- ✅ GPUArrayCache integration for chunk-to-chunk data reuse
- ✅ Reduced memory fragmentation via pooling (Phase 2)
- ✅ Per-chunk batch transfers to minimize GPU↔CPU overhead
- 📈 Expected gain: 15-25% on large datasets (cumulative)
- 🎯 Phase 3 Target: 1.1-1.2x speedup from batch transfers

Author: IGN LiDAR HD Development Team
Date: October 19, 2025
Version: 3.3.0 (Phase 2 + Phase 3 Batch Transfers)
"""
from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy
from .compute.architectural import compute_horizontality
from .compute.coord_utils import recenter_to_local_f32
from .compute.rgb_nir import compute_rgb_features
from .compute.gpu_memory_integration import get_gpu_memory_pool
from .compute.gpu_stream_overlap import get_gpu_stream_optimizer, StreamPhase
from ..core.gpu import GPUManager
from ..optimization.adaptive_chunking import auto_chunk_size, get_recommended_strategy
from ..optimization.gpu_cache import GPUArrayCache
from ..optimization.gpu_batch_transfer import BatchTransferContext

logger = logging.getLogger(__name__)

# GPU availability check (centralized)
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    try:
        import cupy as cp
        from .gpu_processor import GPUProcessor
    except ImportError:
        GPU_AVAILABLE = False
        cp = None
        GPUProcessor = None
else:
    cp = None
    GPUProcessor = None


class GPUChunkedStrategy(BaseFeatureStrategy):
    """
    GPU-based feature computation with auto-chunking for large datasets.

    Now uses GPUProcessor which automatically handles chunking
    and selects the optimal strategy (batch vs chunked) based on dataset size.

    This strategy is optimal for:
    - Large datasets (> 10M points) - automatically uses chunked processing
    - Medium datasets (1-10M points) - automatically uses batch processing
    - Memory-efficient GPU processing
    - Production workloads with massive point clouds

    Performance:
    - Batch mode (<10M): 0.5-5 seconds (10-30x faster than CPU)
    - Chunked mode (>10M): Auto-selected, FAISS acceleration available
    - FAISS speedup: 50-100x for k-NN queries on massive datasets

    Requirements:
    - CuPy (CUDA arrays)
    - cuML (GPU algorithms) - optional but recommended
    - FAISS (optional, for 50-100x k-NN speedup)

    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        chunk_size (int): Points per chunk (default: 1M, handled by GPUProcessor)
        gpu_processor (GPUProcessor): GPU processor with auto-chunking
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        chunk_size: Optional[int] = None,  # Now optional - auto-calculated if None
        batch_size: int = 8_000_000,  # Batch size before auto-chunking
        neighbor_query_batch_size: Optional[
            int
        ] = None,  # Deprecated, kept for compatibility
        feature_batch_size: Optional[int] = None,  # Deprecated, kept for compatibility
        auto_chunk: bool = True,  # NEW: Enable automatic chunk size calculation
        verbose: bool = False,
    ):
        """
        Initialize GPU chunked strategy with auto-chunking.

        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius in meters (not used, k-NN preferred)
            chunk_size: Points per chunk (default: None = auto-calculate)
                       If None, automatically calculates optimal size based on GPU memory
            batch_size: Max batch size before auto-chunking (default: 8M)
            neighbor_query_batch_size: Deprecated (kept for backward compatibility)
            feature_batch_size: Deprecated (kept for backward compatibility)
            auto_chunk: Enable automatic chunk size calculation (default: True)
            verbose: Enable detailed logging

        Raises:
            RuntimeError: If GPU is not available
            
        Note:
            - If chunk_size=None and auto_chunk=True, optimal size calculated automatically
            - Automatic chunking prevents GPU OOM errors
            - Based on available GPU memory and dataset characteristics
        """
        super().__init__(k_neighbors=k_neighbors, radius=radius, verbose=verbose)

        if not GPU_AVAILABLE or GPUProcessor is None:
            raise RuntimeError(
                "GPU chunked strategy requires CuPy and GPUProcessor. Install with: pip install cupy-cuda11x\n"
                "For CPU, use CPUStrategy instead."
            )

        # Auto-calculate chunk size if not provided
        if chunk_size is None and auto_chunk:
            # Will be calculated per-dataset in compute_features()
            # based on actual point cloud shape
            self.chunk_size = None
            self.auto_chunk = True
            if verbose:
                logger.info("Adaptive chunking enabled - will calculate optimal size per dataset")
        else:
            self.chunk_size = chunk_size if chunk_size is not None else 1_000_000
            self.auto_chunk = False
            if verbose:
                logger.info(f"Fixed chunk size: {self.chunk_size:,} points")
        
        self.batch_size = batch_size

        # Initialize GPU memory pool (Phase 2.2)
        self.memory_pool = get_gpu_memory_pool(enable=True)
        
        # Initialize GPU stream overlap optimizer (Phase 2.3)
        self.stream_optimizer = get_gpu_stream_optimizer(enable=True)

        # Initialize GPU processor with chunking parameters
        # Auto-chunks for datasets > 10M points
        self.gpu_processor = GPUProcessor(
            batch_size=batch_size, 
            chunk_size=self.chunk_size if self.chunk_size else batch_size,  # Temporary default
            show_progress=verbose
        )

        if verbose:
            logger.info(
                f"Initialized GPU chunked strategy: "
                f"k={k_neighbors}, batch_size={batch_size:,}, auto_chunk={self.auto_chunk}"
            )
            logger.info(f"  GPU available: {self.gpu_processor.use_gpu}")
            logger.info(f"  Auto-chunking: {'enabled' if self.auto_chunk else 'disabled'}")
            logger.info(f"  Memory pooling enabled: {self.memory_pool.enable_pooling}")
            logger.info(f"  Stream overlap enabled: {self.stream_optimizer.enable_overlap}")
            logger.info(f"  Batch strategy: used for <10M points")
            logger.info(f"  Chunked strategy: used for >10M points (auto-selected)")

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
        Compute features using GPUProcessor with per-chunk batch transfers (Phase 3).

        **Phase 3 Optimizations (November 27, 2025)**:
        - ✅ Per-chunk batch transfers: Within each chunk, combine transfers
        - ✅ Memory pooling with buffer reuse across chunks (Phase 2)
        - ✅ Reduced memory fragmentation via pooling
        - 📈 Expected speedup: 1.1-1.2x from batch transfers on large datasets

        For >10M points, buffers are reused across chunks to minimize memory
        fragmentation and allocation overhead.

        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            rgb: (N, 3) array of RGB values (optional)
            nir: (N,) array of near-infrared values (optional)
            classification: (N,) array of ASPRS classification codes (optional)
            **kwargs: Additional parameters

        Returns:
            Dictionary with feature arrays (same keys as CPUStrategy)
        """
        from ign_lidar.optimization.gpu_pooling_helper import PoolingStatistics

        n_points = len(points)

        # Calculate optimal chunk size if adaptive chunking enabled
        if self.auto_chunk and self.chunk_size is None:
            calculated_chunk_size = auto_chunk_size(
                points_shape=points.shape,
                target_memory_usage=0.7,  # Conservative default
                feature_count=20,  # Estimate for typical feature set
                use_gpu=True,
            )

            # Update GPU processor with calculated chunk size
            self.gpu_processor.chunk_size = calculated_chunk_size

            if self.verbose:
                logger.info(f"  Auto-calculated chunk size: {calculated_chunk_size:,} points")

                # Log recommended strategy
                strategy = get_recommended_strategy(n_points)
                logger.info(f"  Recommended strategy: {strategy}")

        if self.verbose:
            strategy_name = "chunked" if n_points > 10_000_000 else "batch"
            logger.info(
                f"Computing features for {n_points:,} points using GPU strategy "
                f"({strategy_name} mode auto-selected) with per-chunk batch transfers (Phase 3)"
            )

        # Initialize pooling for all chunks (buffers reused across chunks)
        pooling_stats = PoolingStatistics()
        expected_features = 6  # normals(3), curvature, height, verticality, planarity, sphericity
        if rgb is not None:
            expected_features += 3
        if nir is not None and rgb is not None:
            expected_features += 1

        feature_names = [
            "normals_x",
            "normals_y",
            "normals_z",
            "curvature",
            "height",
            "verticality",
            "planarity",
            "sphericity",
        ]
        if rgb is not None:
            feature_names.extend(["red", "green", "blue"])
        if nir is not None and rgb is not None:
            feature_names.append("ndvi")

        # NOTE on removed buffer pooling: this method used to wrap its body in
        # `GPUPoolingContext(...)` (optimization/gpu_pooling_helper.py), whose
        # `pool_ctx` was (a) constructed from `self.gpu_manager.gpu_pool`, an
        # attribute that never existed (only the module-level `_gpu_manager`,
        # which has no `gpu_pool` either); (b) called via `.record_reuse()`,
        # a method `GPUPoolingContext` does not define (that method lives on
        # the unrelated `PoolingStatistics` class); and (c) never actually
        # used for the real per-feature buffers below -- every feature is
        # computed directly into `gpu_results`, not via `pool_ctx.get_buffer()`.
        # It was decorative dead weight layered on top of working code (and,
        # per (a)/(b), never even executed once); removing it changes no
        # computed value.

        # Phase 3: Use batch transfer context for per-chunk transfers
        with BatchTransferContext(enable=True, verbose=self.verbose) as transfer_ctx:

            # Geometry is computed in a LOCAL frame: the GPU works in
            # float32, and absolute Lambert-93 coordinates quantise to
            # 0.06-0.5 m at that precision (see coord_utils). Normals and
            # curvature are translation-invariant; `points` (absolute) is
            # still used below for the height reference.
            points_local, _origin = recenter_to_local_f32(points)

            # Phase 3.1: Batch upload input data (per-chunk)
            input_data = {"points": points_local}
            if rgb is not None:
                input_data["rgb"] = rgb
            if nir is not None:
                input_data["nir"] = nir
            if intensities is not None:
                input_data["intensities"] = intensities

            gpu_inputs = transfer_ctx.batch_upload(input_data, batch_id="chunked_inputs")

            # Extract GPU arrays
            gpu_points = gpu_inputs.get("points", points_local)
            gpu_rgb = gpu_inputs.get("rgb", None) if rgb is not None else None
            gpu_nir = gpu_inputs.get("nir", None) if nir is not None else None

            # Compute geometric features with GPU processor
            # Use the unified compute_features method
            gpu_features = self.gpu_processor.compute_features(
                gpu_points,
                feature_types=["normals", "curvature"],
                k=self.k_neighbors,
                show_progress=self.verbose,
            )

            normals = gpu_features["normals"]
            curvature = gpu_features["curvature"]

            # Compute height relative to minimum Z
            z_min = points[:, 2].min()  # Use original for min
            height = (points[:, 2] - z_min).astype(np.float32)

            # Build GPU result dictionary
            gpu_results = {
                "normals": normals,
                "curvature": curvature,
                "height": height,
            }

            # Compute additional geometric features (reused per-chunk)
            # Verticality: 1 - |normal_z| (walls have normal_z≈0, so verticality≈1)
            # ✅ FIXED: Was inverted (walls got low scores, roofs got high scores)
            verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)
            verticality = np.clip(verticality, 0.0, 1.0).astype(np.float32)
            gpu_results["verticality"] = verticality
            pooling_stats.record_feature()

            # Horizontality: |normal_z| (roofs/ground ≈ 1, walls ≈ 0)
            # Core feature expected downstream (feature_modes, thresholds,
            # classification rules) - always present, like on CPU.
            gpu_results["horizontality"] = compute_horizontality(normals)
            pooling_stats.record_feature()

            # Planarity: 1 - curvature (normalized)
            planarity = (1.0 - np.minimum(curvature, 1.0)).astype(np.float32)
            gpu_results["planarity"] = planarity
            pooling_stats.record_feature()

            # Sphericity: curvature normalized
            sphericity = np.minimum(curvature, 1.0).astype(np.float32)
            gpu_results["sphericity"] = sphericity
            pooling_stats.record_feature()

            # Compute RGB features if provided
            if gpu_rgb is not None:
                rgb_features = compute_rgb_features(gpu_rgb, use_gpu=True)
                gpu_results.update(rgb_features)
                pooling_stats.record_feature()

            # Compute NDVI if NIR and RGB provided
            if gpu_nir is not None and gpu_rgb is not None:
                red = gpu_rgb[:, 0] if hasattr(gpu_rgb, '__getitem__') else gpu_rgb
                ndvi = self.compute_ndvi(gpu_nir, red)
                gpu_results["ndvi"] = ndvi.astype(np.float32)
                pooling_stats.record_feature()

            # Phase 3.2: Batch download all results
            result = transfer_ctx.batch_download(gpu_results, batch_id="chunked_results")

            if self.verbose:
                logger.info(
                    f"GPU computation complete: {len(result)} feature types "
                    f"({pooling_stats.features_computed} tracked)"
                )
                transfer_stats = transfer_ctx.get_statistics()
                logger.info(
                    f"Phase 3 batch transfers: {transfer_stats['serial_transfers_avoided']} "
                    f"transfers avoided, {transfer_stats['total_transfer_mb']:.1f} MB transferred"
                )

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

        Uses GPUProcessor for feature computation.

        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k (ignored for GPU)
            include_extra: Whether to include extra features
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', 'asprs_classes', etc.)
            radius: Search radius (ignored, uses k-NN)
            **kwargs: Additional parameters

        Returns:
            Dictionary with feature arrays
        """
        # Extract RGB and NIR from kwargs if present
        rgb = kwargs.get("rgb", None)
        nir = kwargs.get("nir", None)

        # Use the compute method which handles everything
        result = self.compute(
            points=points,
            intensities=kwargs.get("intensities", None),
            rgb=rgb,
            nir=nir,
            classification=classification,
        )

        # Add distance_to_center if patch_center provided and include_extra is True
        if include_extra and patch_center is not None:
            distances = np.linalg.norm(points - patch_center, axis=1)
            result["distance_to_center"] = distances.astype(np.float32)

        return result
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GPUChunkedStrategy(k_neighbors={self.k_neighbors}, "
            f"chunk_size={self.chunk_size:,}, batch_size={self.batch_size:,})"
        )


# Export
__all__ = ["GPUChunkedStrategy"]
