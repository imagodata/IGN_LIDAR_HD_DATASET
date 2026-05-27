"""
Structured configuration schema for IGN LiDAR HD.

⚠️  DEPRECATED - WILL BE REMOVED IN v4.0.0 (Q1 2026) ⚠️

This module is deprecated in v3.2+ in favor of ign_lidar.config.Config
and will be COMPLETELY REMOVED in v4.0.0.

════════════════════════════════════════════════════════════════════════════════
                            ⚠️  ACTION REQUIRED  ⚠️
════════════════════════════════════════════════════════════════════════════════

You MUST migrate to the new Config system before v4.0.0 (scheduled Q1 2026).

MIGRATION STEPS:
    1. Run the automatic migration tool:
       $ ign-lidar-hd migrate-config your_config.yaml
    
    2. Update your Python imports:
       # OLD (deprecated):
       from ign_lidar.config.schema import ProcessorConfig, FeaturesConfig
       
       # NEW (v4.0):
       from ign_lidar.config import Config
    
    3. Update your code:
       # OLD:
       processor_config = ProcessorConfig(lod_level='LOD2')
       features_config = FeaturesConfig(mode='lod2')
       
       # NEW:
       config = Config.preset('lod2_buildings')
       # Or: config = Config(mode='lod2', input_dir='...', output_dir='...')
    
    4. Test with v3.9 (includes migration tools and backward compatibility)
    
    5. Upgrade to v4.0 when ready

TIMELINE:
    - NOW (v3.9): Deprecation warnings, migration tool available
    - Q1 2026 (v4.0.0): schema.py REMOVED (breaking change)

HELP & RESOURCES:
    - Migration Guide: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/guides/migration-v3-to-v4/
    - Config v4.0 Docs: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/configuration/
    - GitHub Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
    - Support: Open a GitHub discussion

════════════════════════════════════════════════════════════════════════════════
"""

import warnings

warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATION WARNING  ⚠️\n\n"
    "ign_lidar.config.schema will be REMOVED in v4.0.0 (Q1 2026)\n\n"
    "ACTION REQUIRED: Migrate to ign_lidar.config.Config\n\n"
    "Quick Migration:\n"
    "  1. Run: ign-lidar-hd migrate-config your_config.yaml\n"
    "  2. Update imports: from ign_lidar.config import Config\n"
    "  3. Use: config = Config.preset('lod2_buildings')\n\n"
    "See migration guide: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/guides/migration-v3-to-v4/\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from omegaconf import MISSING


@dataclass
class ProcessorConfig:
    """
    Configuration for the main LiDAR processor.

    Attributes:
        lod_level: Level of Detail classification target ('LOD2' or 'LOD3')
        architecture: Neural network architecture ('pointnet++', 'hybrid', 'octree', 'transformer', 'sparse_conv', 'multi')
        use_gpu: Enable GPU acceleration (requires CuPy)
        num_workers: Number of parallel workers for processing
        patch_size: Size of patches in meters
        patch_overlap: Overlap ratio between patches (0.0-1.0)
        num_points: Target number of points per patch
        augment: Enable data augmentation
        num_augmentations: Number of augmentations per patch
        batch_size: Batch size for processing ('auto' or int)
        prefetch_factor: Number of batches to prefetch
        pin_memory: Pin memory for faster GPU transfer
    """

    lod_level: Literal["LOD2", "LOD3"] = "LOD2"
    architecture: Literal[
        "pointnet++", "hybrid", "octree", "transformer", "sparse_conv", "multi", "ptv3"
    ] = "pointnet++"
    use_gpu: bool = False
    num_workers: int = 4

    # Patch configuration
    patch_size: float = 150.0
    patch_overlap: float = 0.1
    num_points: int = 16384

    # Augmentation
    augment: bool = False
    num_augmentations: int = 3

    # Performance tuning
    batch_size: Union[str, int] = "auto"
    prefetch_factor: int = 2
    pin_memory: bool = False

    # Optional reclassification settings
    reclassification: Optional[dict] = None


@dataclass
class FeaturesConfig:
    """
    Configuration for feature computation.

    Attributes:
        mode: Feature computation mode ('minimal', 'full', 'custom')
        k_neighbors: Number of neighbors for geometric features
        search_radius: Search radius in meters (recommended: 1.0-2.0m)
        include_extra: Include extra features
        use_rgb: Include RGB from IGN orthophotos
        use_infrared: Include near-infrared from IRC
        compute_ndvi: Compute NDVI vegetation index
        sampling_method: Point sampling strategy
        normalize_xyz: Normalize XYZ coordinates to [-1, 1]
        normalize_features: Standardize features (z-score)

        # Multi-scale (v6.2+)
        multi_scale_computation: Enable multi-scale computation
        scales: List of scale configs (name, k_neighbors, radius, weight)
        aggregation_method: Method for combining scales
        variance_penalty_factor: Penalty for high-variance scales
        artifact_detection: Enable artifact detection
        artifact_variance_threshold: Variance threshold for artifacts
        artifact_gradient_threshold: Gradient threshold for artifacts
        auto_suppress_artifacts: Auto-reduce artifact scale weights
        adaptive_scale_selection: Enable adaptive scale per point
        complexity_threshold: Complexity threshold for adaptation
        homogeneity_threshold: Homogeneity threshold for adaptation
        save_scale_quality_metrics: Save quality metrics
        save_selected_scale: Save selected scale index
        reuse_kdtrees_across_scales: Reuse KD-trees
        parallel_scale_computation: Parallel scale computation
        cache_scale_results: Cache intermediate results
    """

    mode: Literal["minimal", "full", "custom"] = "full"
    k_neighbors: int = 20
    search_radius: Optional[float] = None  # Auto-estimate if None, use k_neighbors if 0

    # Feature flags
    include_extra: bool = False
    use_rgb: bool = False
    use_infrared: bool = False
    compute_ndvi: bool = False

    # PointNet++ specific optimizations
    sampling_method: Literal["random", "fps", "grid"] = "random"
    normalize_xyz: bool = False
    normalize_features: bool = False

    # GPU configuration
    gpu_batch_size: int = 1_000_000
    use_gpu_chunked: bool = True

    # Multi-scale configuration (v6.2+)
    multi_scale_computation: bool = False
    scales: Optional[List[Dict[str, Any]]] = None
    aggregation_method: Literal["weighted_average", "variance_weighted", "adaptive"] = (
        "variance_weighted"
    )
    variance_penalty_factor: float = 2.0

    # Artifact detection
    artifact_detection: bool = False
    artifact_variance_threshold: float = 0.15
    artifact_gradient_threshold: float = 0.10
    auto_suppress_artifacts: bool = True

    # Adaptive scale selection
    adaptive_scale_selection: bool = False
    complexity_threshold: float = 0.5
    homogeneity_threshold: float = 0.8

    # Multi-scale output options
    save_scale_quality_metrics: bool = False
    save_selected_scale: bool = False

    # Multi-scale performance optimization
    reuse_kdtrees_across_scales: bool = True
    parallel_scale_computation: bool = False
    cache_scale_results: bool = True

    def __post_init__(self):
        """Validate multi-scale configuration."""
        if self.multi_scale_computation:
            if not self.scales or len(self.scales) < 2:
                raise ValueError(
                    "Multi-scale computation requires at least 2 scales. "
                    "Please provide a list of scale configurations with 'name', 'k_neighbors', "
                    "'search_radius', and 'weight' for each scale."
                )

            # Validate each scale configuration
            required_fields = {"name", "k_neighbors", "search_radius", "weight"}
            for i, scale in enumerate(self.scales):
                missing = required_fields - set(scale.keys())
                if missing:
                    raise ValueError(
                        f"Scale {i} is missing required fields: {missing}. "
                        f"Each scale must have: {required_fields}"
                    )

                # Validate numeric fields
                if scale["k_neighbors"] <= 0:
                    raise ValueError(
                        f"Scale {i} ({scale['name']}): k_neighbors must be > 0"
                    )
                if scale["search_radius"] <= 0:
                    raise ValueError(
                        f"Scale {i} ({scale['name']}): search_radius must be > 0"
                    )
                if scale["weight"] < 0:
                    raise ValueError(
                        f"Scale {i} ({scale['name']}): weight must be >= 0"
                    )

            # Validate aggregation method compatibility
            if (
                self.aggregation_method == "adaptive"
                and not self.adaptive_scale_selection
            ):
                raise ValueError(
                    "aggregation_method='adaptive' requires adaptive_scale_selection=true"
                )

            # Warn about performance implications
            if self.parallel_scale_computation and len(self.scales) > 3:
                import warnings

                warnings.warn(
                    f"Parallel scale computation with {len(self.scales)} scales may use "
                    f"significant memory. Consider serial computation or reducing scales.",
                    UserWarning,
                )


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing (outlier removal, etc.).

    Attributes:
        enabled: Enable preprocessing
        sor_k: Number of neighbors for Statistical Outlier Removal
        sor_std: Standard deviation threshold for SOR
        ror_radius: Search radius for Radius Outlier Removal (meters)
        ror_neighbors: Minimum neighbors for ROR
        voxel_enabled: Enable voxel downsampling
        voxel_size: Voxel size for downsampling (meters)
    """

    enabled: bool = False

    # Statistical Outlier Removal
    sor_k: int = 12
    sor_std: float = 2.0

    # Radius Outlier Removal
    ror_radius: float = 1.0
    ror_neighbors: int = 4

    # Voxel downsampling
    voxel_enabled: bool = False
    voxel_size: float = 0.1


@dataclass
class StitchingConfig:
    """
    Configuration for tile stitching (boundary-aware processing).

    Attributes:
        enabled: Enable tile stitching
        buffer_size: Buffer zone size in meters
        auto_detect_neighbors: Automatically detect neighbor tiles
        auto_download_neighbors: Automatically download missing neighbor tiles from IGN WFS
        cache_enabled: Cache loaded tiles to avoid re-reading
    """

    enabled: bool = False
    buffer_size: float = 10.0
    auto_detect_neighbors: bool = True
    auto_download_neighbors: bool = False
    cache_enabled: bool = True


@dataclass
class OutputConfig:
    """
    Configuration for output formats and saving.

    Attributes:
        format: Output format ('npz', 'hdf5', 'torch', 'laz', 'all')
        processing_mode: Processing mode - 'patches_only' (default), 'both', or 'enriched_only'
                        - 'patches_only': Create ML patches only (default, fastest for training)
                        - 'both': Create both patches and enriched LAZ files
                        - 'enriched_only': Only create enriched LAZ (fastest for GIS)
        save_stats: Save processing statistics
        save_metadata: Save patch metadata
        compression: Compression level (0-9, None for no compression)
        skip_existing: Skip tiles that have already been processed (default True)
    """

    format: Literal["npz", "hdf5", "torch", "laz", "all"] = "npz"
    processing_mode: Literal["patches_only", "both", "enriched_only"] = "patches_only"
    save_stats: bool = True
    save_metadata: bool = True
    compression: Optional[int] = None
    skip_existing: bool = True


@dataclass
class BBoxConfig:
    """
    Optional bounding box configuration for spatial filtering.

    Attributes:
        xmin: Minimum X coordinate
        ymin: Minimum Y coordinate
        xmax: Maximum X coordinate
        ymax: Maximum Y coordinate
    """

    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None

    def to_tuple(self) -> Optional[tuple]:
        """Convert to tuple format (xmin, ymin, xmax, ymax)."""
        if all(v is not None for v in [self.xmin, self.ymin, self.xmax, self.ymax]):
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        return None


@dataclass
class IGNLiDARConfig:
    """
    Root configuration for IGN LiDAR HD processing.

    This is the main configuration class that composes all sub-configurations.
    Used with Hydra for hierarchical configuration management.

    Example:
        >>> cfg = IGNLiDARConfig(
        ...     input_dir="data/raw",
        ...     output_dir="data/patches",
        ...     processor=ProcessorConfig(use_gpu=True)
        ... )
    """

    # Sub-configurations (populated from YAML files)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    bbox: BBoxConfig = field(default_factory=BBoxConfig)

    # I/O paths (required - must be provided by user)
    input_dir: str = MISSING
    output_dir: str = MISSING

    # Global settings
    num_workers: int = 4
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Hydra configuration composition
    # This defines which config groups to load by default
    defaults: List[str] = field(
        default_factory=lambda: [
            {"processor": "default"},
            {"features": "full"},
            {"preprocess": "default"},
            {"stitching": "disabled"},
            {"output": "default"},
            "_self_",
        ]
    )

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate patch configuration
        if self.processor.patch_size <= 0:
            raise ValueError("patch_size must be > 0")

        if not 0 <= self.processor.patch_overlap < 1:
            raise ValueError("patch_overlap must be in [0, 1)")

        if self.processor.num_points <= 0:
            raise ValueError("num_points must be > 0")

        # Validate feature configuration
        if self.features.k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")

        # Validate preprocessing
        if self.preprocess.enabled:
            if self.preprocess.sor_k <= 0:
                raise ValueError("sor_k must be > 0")
            if self.preprocess.ror_radius <= 0:
                raise ValueError("ror_radius must be > 0")

        # Validate stitching
        if self.stitching.enabled:
            if self.stitching.buffer_size <= 0:
                raise ValueError("buffer_size must be > 0")

            # Buffer size should be >= 2 * radius used for features
            # Estimate radius from k_neighbors (rough heuristic)
            estimated_radius = self.features.k_neighbors * 0.5  # meters
            if self.stitching.buffer_size < 2 * estimated_radius:
                import logging

                logging.warning(
                    f"buffer_size ({self.stitching.buffer_size}m) should be "
                    f">= 2 * feature_radius (~{2*estimated_radius:.1f}m) "
                    f"for optimal boundary quality"
                )
