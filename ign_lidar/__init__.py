"""
IGN LiDAR HD Dataset Processing Library

A Python library for processing IGN LiDAR HD data into machine learning-ready datasets
with building LOD (Level of Detail) classification support.

Version 3.3.3 - Gap Detection Enhancement:
- Automatic gap/void detection in building perimeters
- Angular sector analysis (36 sectors = 10° each)
- Gap-based adaptive buffer adjustment
- Quality metrics for building detection
- Directional gap identification (N, NE, E, SE, S, SW, W, NW)
- Export of problematic buildings for manual review

Version 3.5.0 - Package Harmonization & Documentation Consolidation:
- Consolidated and harmonized all version references to 3.5.0
- Updated CHANGELOG with comprehensive release history
- Harmonized README, Docusaurus intro, and documentation
- Cleaned up configuration file examples
- Improved consistency across all package files
- Clear documentation structure for users

Version 3.4.1 - FAISS GPU Memory Optimization:
- Dynamic VRAM detection and adaptive GPU usage for FAISS k-NN
- Automatic Float16 (FP16) precision for large datasets (>50M points)
- Smart memory calculation: query results + index + temp storage
- Adaptive threshold: 80% of VRAM limit (vs hardcoded 15M point limit)
- RTX 4080 SUPER (16GB): 72M point dataset now uses GPU (was CPU-only)
- Expected speedup: 10-50x faster k-NN queries
- Supports 100M+ points on 16GB GPUs with FP16

Version 3.4.0 - GPU Optimizations & Road Classification:
- GPU-accelerated operations (k-NN, STRtree spatial indexing)
- FAISS GPU integration for ultra-fast k-NN queries
- WFS optimization with caching and parallel processing
- Road classification from BD TOPO implementation
- Performance improvements across the pipeline
- New benchmarking and evaluation tools

All features now work with the new configuration architecture.
"""

__version__ = "4.0.1"
__author__ = "imagodata"
__email__ = "simon.ducournau@gmail.com"

# ============================================================================
# v3.2+ API (Recommended - Simplified Configuration)
# ============================================================================

# Configuration
from .config import AdvancedConfig, Config, FeatureConfig

# Classification
from .core.classification import BaseClassifier, ClassificationResult

# Core processing modules
from .core.processor import LiDARProcessor
from .core import ground_truth

# Feature extraction
from .features import (
    compute_all_features_optimized,
    compute_curvature,
    compute_normals,
    extract_geometric_features,
)

# Preprocessing
from .preprocessing import (
    add_infrared_to_patch,
    add_rgb_to_patch,
    augment_tile_with_infrared,
    augment_tile_with_rgb,
    preprocess_point_cloud,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)

# ============================================================================
# v3.0+ IMPORTS (Legacy but still supported)
# ============================================================================


# Ground Truth (WFS) - Import on demand to avoid dependency issues
try:
    from .io.wfs_ground_truth import (
        IGNGroundTruthFetcher,
        IGNWFSConfig,
        fetch_ground_truth_for_tile,
        generate_patches_with_ground_truth,
    )
except ImportError:
    # shapely/geopandas not installed
    IGNWFSConfig = None
    IGNGroundTruthFetcher = None
    fetch_ground_truth_for_tile = None
    generate_patches_with_ground_truth = None

# Configuration (Hydra) - Import on demand to avoid dependency issues
try:
    from .config.schema import (
        FeaturesConfig,
        IGNLiDARConfig,
        OutputConfig,
        PreprocessConfig,
        ProcessorConfig,
        StitchingConfig,
    )
except ImportError:
    # Hydra dependencies not installed
    IGNLiDARConfig = None
    ProcessorConfig = None
    FeaturesConfig = None
    PreprocessConfig = None
    StitchingConfig = None
    OutputConfig = None

# ============================================================================
# BACKWARD COMPATIBILITY IMPORTS (Legacy - Still Supported)
# ============================================================================

# v3.1: classification schema (consolidates asprs_classes.py + classes.py)
from .classification_schema import LOD2_CLASSES  # Backward compatibility
from .classification_schema import LOD3_CLASSES  # Backward compatibility
from .classification_schema import (
    ASPRS_CLASS_NAMES,
    ASPRS_TO_LOD2,
    ASPRS_TO_LOD3,
    BRIDGE_NATURE_TO_ASPRS,
    CEMETERY_NATURE_TO_ASPRS,
    PARKING_NATURE_TO_ASPRS,
    POWER_LINE_NATURE_TO_ASPRS,
    RAILWAY_NATURE_TO_ASPRS,
    SPORTS_NATURE_TO_ASPRS,
    ASPRSClass,
    ClassificationMode,
    LOD2Class,
    LOD3Class,
    get_class_color,
    get_class_name,
    get_classification_for_bridge,
    get_classification_for_building,
    get_classification_for_cemetery,
    get_classification_for_parking,
    get_classification_for_power_line,
    get_classification_for_railway,
    get_classification_for_road,
    get_classification_for_sports,
    get_classification_for_vegetation,
    get_classification_for_water,
)

# Root level modules (unchanged location)
from .downloader import IGNLiDARDownloader

# v4.0: Removed backward compatibility for deprecated modules:
# - ign_lidar.classes (use ign_lidar.classification_schema)
# - ign_lidar.asprs_classes (use ign_lidar.classification_schema)

# Reorganized modules - backward compatibility imports
# Core utilities (moved to core/)
from .core import (
    AdaptiveMemoryManager,
    GPUMemoryError,
    MemoryConfig,
    MemoryPressureError,
    PerformanceMonitor,
    ProcessingError,
)

# Dataset utilities (moved to datasets/)
from .datasets import (
    STRATEGIC_LOCATIONS,
    WORKING_TILES,
    get_tiles_by_environment,
    get_tiles_by_priority,
    get_tiles_by_region,
)

# Feature extraction - use modern core modules directly
# Feature utilities (moved to features/)
from .features import (
    ARCHITECTURAL_STYLES,
    STYLE_NAME_TO_ID,
    compute_architectural_style_features,
    compute_curvature,
    compute_normals,
    extract_geometric_features,
    get_patch_architectural_style,
    get_tile_architectural_style,
)

# IO utilities (moved to io/)
from .io import MetadataManager, simplify_for_qgis

# Preprocessing utilities (moved to preprocessing/)
from .preprocessing import analyze_tile, augment_raw_points, extract_patches

# Backward compatibility for moved modules - these imports should be available
# at the root level for legacy code
try:
    # Import core modules at root level for backward compatibility
    # Make them available as if imported from root
    import sys
    import types

    from .core.processor import LiDARProcessor as processor_LiDARProcessor
    from .core.tile_stitcher import TileStitcher

    # Create processor module for backward compatibility
    processor_module = types.ModuleType("ign_lidar.processor")
    processor_module.LiDARProcessor = processor_LiDARProcessor
    sys.modules["ign_lidar.processor"] = processor_module

    # Create tile_stitcher module for backward compatibility
    tile_stitcher_module = types.ModuleType("ign_lidar.tile_stitcher")
    tile_stitcher_module.TileStitcher = TileStitcher
    sys.modules["ign_lidar.tile_stitcher"] = tile_stitcher_module

except ImportError:
    pass

__all__ = [
    # ========================================================================
    # v3.2+ Main API (NEW - Simplified)
    # ========================================================================
    "Config",
    "FeatureConfig",
    "AdvancedConfig",
    "BaseClassifier",
    "ClassificationResult",
    # ========================================================================
    # Core v3.0+
    # ========================================================================
    # ========== Core v2.0 ==========
    # Processor
    "LiDARProcessor",
    # Ground Truth v2.0
    "ground_truth",
    # Features
    "compute_normals",
    "compute_curvature",
    "extract_geometric_features",
    "compute_all_features_optimized",
    # Preprocessing
    "statistical_outlier_removal",
    "radius_outlier_removal",
    "voxel_downsample",
    "preprocess_point_cloud",
    "add_rgb_to_patch",
    "augment_tile_with_rgb",
    "add_infrared_to_patch",
    "augment_tile_with_infrared",
    # Ground Truth (WFS)
    "IGNWFSConfig",
    "IGNGroundTruthFetcher",
    "fetch_ground_truth_for_tile",
    "generate_patches_with_ground_truth",
    # Configuration
    "IGNLiDARConfig",
    "ProcessorConfig",
    "FeaturesConfig",
    "PreprocessConfig",
    "StitchingConfig",
    "OutputConfig",
    # ========== Root Level (Core Package) ==========
    # Downloader
    "IGNLiDARDownloader",
    # Classification
    "LOD2_CLASSES",
    "LOD3_CLASSES",
    "ASPRSClass",
    "ASPRS_CLASS_NAMES",
    "ClassificationMode",
    "get_classification_for_building",
    "get_classification_for_road",
    "get_classification_for_vegetation",
    "get_classification_for_water",
    "get_classification_for_railway",
    "get_classification_for_sports",
    "get_classification_for_cemetery",
    "get_classification_for_power_line",
    "get_classification_for_parking",
    "get_classification_for_bridge",
    "get_class_name",
    "get_class_color",
    "RAILWAY_NATURE_TO_ASPRS",
    "SPORTS_NATURE_TO_ASPRS",
    "CEMETERY_NATURE_TO_ASPRS",
    "POWER_LINE_NATURE_TO_ASPRS",
    "PARKING_NATURE_TO_ASPRS",
    "BRIDGE_NATURE_TO_ASPRS",
    # ========== Reorganized Modules (Backward Compatibility) ==========
    # Core utilities
    "AdaptiveMemoryManager",
    "MemoryConfig",
    "PerformanceMonitor",
    "ProcessingError",
    "GPUMemoryError",
    "MemoryPressureError",
    "TileStitcher",  # Backward compatibility for tile stitching
    # Feature utilities
    "ARCHITECTURAL_STYLES",
    "STYLE_NAME_TO_ID",
    "get_tile_architectural_style",
    "get_patch_architectural_style",
    "compute_architectural_style_features",
    # Dataset utilities
    "STRATEGIC_LOCATIONS",
    "WORKING_TILES",
    "get_tiles_by_environment",
    "get_tiles_by_priority",
    "get_tiles_by_region",
    # IO utilities
    "MetadataManager",
    "simplify_for_qgis",
    # Preprocessing utilities
    "augment_raw_points",
    "extract_patches",
    "analyze_tile",
]
