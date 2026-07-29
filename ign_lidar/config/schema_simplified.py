"""
Simplified configuration schema for IGN LiDAR HD (v3.0).

DEPRECATED: This module is deprecated in v3.2+ in favor of ign_lidar.config.Config

Both schema.py and schema_simplified.py have been replaced with a single
Config class that is even simpler and more intuitive.

Migration Guide:
    # Old (v3.0-3.1, deprecated)
    from ign_lidar.config.schema_simplified import ProcessingConfig
    config = ProcessingConfig(lod_level='LOD2')

    # New (v3.2+, recommended)
    from ign_lidar.config import Config
    config = Config.preset('lod2_buildings')

This module will be REMOVED in v4.0.0.

For migration help:
    ign-lidar migrate-config old_config.yaml

Author: Refactoring Team
Date: October 16, 2025
"""

import warnings

warnings.warn(
    "ign_lidar.config.schema_simplified is deprecated and will be removed "
    "in v4.0.0. Use ign_lidar.config.Config instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from omegaconf import MISSING

# ============================================================================
# Core Configuration
# ============================================================================


@dataclass
class ProcessingConfig:
    """
    Core processing configuration - simplified.

    Key changes from v2.x:
    - Merged processor.* and features.* namespaces where logical
    - Removed redundant gpu_batch_size (now auto-calculated)
    - Simplified augmentation parameters
    - Clearer defaults
    """

    # Processing mode
    lod_level: Literal["ASPRS", "LOD2", "LOD3"] = "LOD2"
    mode: Literal["patches_only", "both", "enriched_only", "reclassify_only"] = (
        "patches_only"
    )
    architecture: Literal[
        "pointnet++", "hybrid", "octree", "transformer", "sparse_conv", "multi"
    ] = "pointnet++"

    # Performance
    num_workers: int = 4
    use_gpu: bool = False

    # Patch extraction
    patch_size: float = 150.0
    patch_overlap: float = 0.1
    num_points: int = 16384

    # Augmentation (simplified)
    augment: bool = False
    num_augmentations: int = 3

    # Optional reclassification
    reclassification: Optional[dict] = None


@dataclass
class FeatureConfig:
    """
    Feature computation configuration - simplified.

    Key changes:
    - Added 'mode' parameter to select feature set (asprs_classes, minimal, lod2, lod3, full)
    - Simplified RGB/NIR flags
    - Clearer neighbor search parameters
    - Removed rarely-used normalization flags

    Feature Modes:
    - 'asprs_classes': Optimized for ASPRS classification (~15 features, lightweight)
    - 'minimal': Ultra-fast basic features (~8 features)
    - 'lod2': Essential features for LOD2 building detection (~17 features)
    - 'lod3': Complete features for LOD3 architectural modeling (~43 features)
    - 'full': All available features (~45 features)
    """

    # Feature set selection
    mode: Literal["asprs_classes", "minimal", "lod2", "lod3", "full", "custom"] = (
        "asprs_classes"
    )

    # Geometric features
    k_neighbors: int = 20
    search_radius: Optional[float] = None  # Auto if None, overrides k_neighbors if set

    # Spectral features (simplified)
    use_rgb: bool = False
    use_nir: bool = False
    compute_ndvi: bool = False

    # Advanced features (optional)
    include_extra: bool = False  # Height stats, verticality, etc.

    # Height above ground: 'ground_plane' (default, per-tile/patch min Z of
    # ground points), 'min_z', or 'dtm' (RGE ALTI/LiDAR HD MNT, more accurate
    # on sloped terrain — see FeaturesConfig in schema.py for details)
    height_method: Literal["ground_plane", "min_z", "dtm"] = "ground_plane"
    dtm_cache_dir: Optional[str] = None
    dtm_local_dir: Optional[str] = None
    dtm_prefer_lidar_hd: bool = True

    # GPU settings (auto-tuned if not specified)
    gpu_batch_size: Optional[int] = None  # Auto-calculated based on available memory
    use_gpu_chunked: bool = True


@dataclass
class PreprocessConfig:
    """
    Preprocessing configuration - simplified.

    Most users don't need this, so it's optional and disabled by default.
    """

    enabled: bool = False

    # Statistical Outlier Removal
    sor_k: int = 12
    sor_std: float = 2.0

    # Radius Outlier Removal
    ror_radius: float = 1.0
    ror_neighbors: int = 4


@dataclass
class DataSourceConfig:
    """
    Multi-source data enrichment configuration - simplified.

    Key changes:
    - Flattened structure (no more nested enabled/features/parameters)
    - Clearer parameter names
    - Sensible defaults
    """

    # BD TOPO (topographic database)
    bd_topo_enabled: bool = False
    bd_topo_buildings: bool = True
    bd_topo_roads: bool = True
    bd_topo_water: bool = True
    bd_topo_vegetation: bool = True
    bd_topo_cache_dir: str = "cache/bd_topo"

    # BD Forêt (forest database)
    bd_foret_enabled: bool = False
    bd_foret_cache_dir: str = "cache/bd_foret"

    # RPG (agricultural parcels)
    rpg_enabled: bool = False
    rpg_year: int = 2024
    rpg_cache_dir: str = "cache/rpg"

    # Cadastre (land parcels)
    cadastre_enabled: bool = False
    cadastre_cache_dir: str = "cache/cadastre"


@dataclass
class OutputConfig:
    """
    Output configuration - simplified.

    Key changes:
    - Removed redundant format options
    - Clearer naming
    """

    format: Literal["npz", "hdf5", "torch", "laz"] = "npz"
    save_stats: bool = True
    skip_existing: bool = True
    compression: Optional[int] = None


@dataclass
class BBoxConfig:
    """
    Spatial filtering (unchanged - simple and clear).
    """

    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None


# ============================================================================
# Root Configuration
# ============================================================================


@dataclass
class IGNLiDARConfig:
    """
    Root configuration for IGN LiDAR HD processing (v3.0 simplified).

    Key improvements:
    - Flatter structure (fewer nested levels)
    - Clearer parameter grouping
    - Removed redundant parameters
    - Better defaults

    Migration from v2.x:
    - config.processor.* → config.processing.*
    - config.features.* → config.features.*
    - config.data_sources.bd_topo.enabled → config.data_sources.bd_topo_enabled
    - config.output.processing_mode → config.processor.processing_mode

    Example:
        >>> cfg = IGNLiDARConfig(
        ...     input_dir="data/raw",
        ...     output_dir="data/processed",
        ...     processing=ProcessingConfig(use_gpu=True, lod_level="LOD3")
        ... )
    """

    # Required I/O paths
    input_dir: str = MISSING
    output_dir: str = MISSING

    # Sub-configurations
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    bbox: BBoxConfig = field(default_factory=BBoxConfig)

    # Global settings
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate patch configuration
        if self.processing.patch_size <= 0:
            raise ValueError("patch_size must be > 0")

        if not 0 <= self.processing.patch_overlap < 1:
            raise ValueError("patch_overlap must be in [0, 1)")

        if self.processing.num_points <= 0:
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


# ============================================================================
# Backward Compatibility Helpers
# ============================================================================


def migrate_config_v2_to_v3(old_config: dict) -> IGNLiDARConfig:
    """
    Migrate v2.x configuration to v3.0 simplified structure.

    Args:
        old_config: v2.x configuration dictionary

    Returns:
        v3.0 IGNLiDARConfig object

    Example:
        >>> old_cfg = load_yaml("old_config.yaml")
        >>> new_cfg = migrate_config_v2_to_v3(old_cfg)
    """
    import warnings

    warnings.warn(
        "Migrating v2.x configuration to v3.0. "
        "Please update your config files to the new format.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Extract old structure
    processor = old_config.get("processor", {})
    features = old_config.get("features", {})
    data_sources = old_config.get("data_sources", {})
    output_cfg = old_config.get("output", {})

    # Build new structure
    new_config = IGNLiDARConfig(
        input_dir=old_config.get("input_dir", MISSING),
        output_dir=old_config.get("output_dir", MISSING),
        processing=ProcessingConfig(
            lod_level=processor.get("lod_level", "LOD2"),
            mode=output_cfg.get("processing_mode", "patches_only"),
            architecture=processor.get("architecture", "pointnet++"),
            num_workers=processor.get("num_workers", 4),
            use_gpu=processor.get("use_gpu", False),
            patch_size=processor.get("patch_size", 150.0),
            patch_overlap=processor.get("patch_overlap", 0.1),
            num_points=processor.get("num_points", 16384),
            augment=processor.get("augment", False),
            num_augmentations=processor.get("num_augmentations", 3),
            reclassification=processor.get("reclassification"),
        ),
        features=FeatureConfig(
            k_neighbors=features.get("k_neighbors", 20),
            search_radius=features.get("search_radius"),
            use_rgb=features.get("use_rgb", False),
            use_nir=features.get("use_infrared", False),
            compute_ndvi=features.get("compute_ndvi", False),
            include_extra=features.get("include_extra", False),
            gpu_batch_size=features.get("gpu_batch_size"),
            use_gpu_chunked=features.get("use_gpu_chunked", True),
        ),
        preprocess=PreprocessConfig(
            enabled=old_config.get("preprocess", {}).get("enabled", False),
        ),
        data_sources=DataSourceConfig(
            bd_topo_enabled=data_sources.get("bd_topo", {}).get("enabled", False),
            bd_topo_buildings=data_sources.get("bd_topo", {})
            .get("features", {})
            .get("buildings", True),
            bd_topo_roads=data_sources.get("bd_topo", {})
            .get("features", {})
            .get("roads", True),
            bd_topo_water=data_sources.get("bd_topo", {})
            .get("features", {})
            .get("water", True),
            bd_topo_vegetation=data_sources.get("bd_topo", {})
            .get("features", {})
            .get("vegetation", True),
            bd_foret_enabled=data_sources.get("bd_foret", {}).get("enabled", False),
            rpg_enabled=data_sources.get("rpg", {}).get("enabled", False),
            rpg_year=data_sources.get("rpg", {}).get("year", 2024),
            cadastre_enabled=data_sources.get("cadastre", {}).get("enabled", False),
        ),
        output=OutputConfig(
            format=output_cfg.get("format", "npz"),
            save_stats=output_cfg.get("save_stats", True),
            skip_existing=output_cfg.get("skip_existing", True),
            compression=output_cfg.get("compression"),
        ),
        bbox=BBoxConfig(
            xmin=old_config.get("bbox", {}).get("xmin"),
            ymin=old_config.get("bbox", {}).get("ymin"),
            xmax=old_config.get("bbox", {}).get("xmax"),
            ymax=old_config.get("bbox", {}).get("ymax"),
        ),
        verbose=old_config.get("verbose", True),
        log_level=old_config.get("log_level", "INFO"),
    )

    return new_config


def get_config_value(config: Union[IGNLiDARConfig, dict], key_path: str, default=None):
    """
    Helper to access config values with dot notation for backward compatibility.

    Args:
        config: Configuration object or dictionary
        key_path: Dot-separated path (e.g., 'processing.use_gpu')
        default: Default value if path not found

    Returns:
        Configuration value or default

    Example:
        >>> use_gpu = get_config_value(cfg, 'processing.use_gpu', False)
    """
    parts = key_path.split(".")
    value = config

    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)

        if value is None:
            return default

    return value if value is not None else default
