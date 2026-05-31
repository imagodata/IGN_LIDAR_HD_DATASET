"""
Main LiDAR Processing Class
"""

import gc
import logging
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import laspy
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..classification_schema import ASPRS_TO_LOD2, ASPRS_TO_LOD3

# Dataset manager for ML dataset creation with train/val/test splits
from ..datasets.dataset_manager import DatasetConfig, DatasetManager
from ..features.architectural_styles import get_architectural_style_id

# Phase 4.3: FeatureOrchestrator V5 (consolidated)
from ..features.orchestrator import FeatureOrchestrator
from ..io.metadata import MetadataManager

# Classification module (consolidated in v3.1.0, renamed in v3.3.0)
from .classification import Classifier, refine_classification

# Import refactored modules from classification package
# Note: FeatureManager has been replaced by FeatureOrchestrator in Phase 4.3
from .classification.config_validator import ConfigValidator

# Phase 3.4: Tile processing modules
from .classification.io import (
    TileLoader,
    save_patch_hdf5,
    save_patch_laz,
    save_patch_multi_format,
    save_patch_npz,
    save_patch_torch,
)
from .classification.patch_extractor import (
    AugmentationConfig,
    PatchConfig,
    extract_and_augment_patches,
    format_patch_for_architecture,
)

# Reclassification module
from .classification.reclassifier import (
    Reclassifier,
    reclassify_tile,
)
from .classification_applier import ClassificationApplier
from .gpu_context import disable_gpu_for_multiprocessing

# Optimization factory for intelligent strategy selection
from .optimization_factory import auto_optimize_config, optimization_factory
from .output_writer import OutputWriter
from .patch_extractor import PatchExtractor
from .processing_metadata import ProcessingMetadata

# Phase 2: Refactored components (v3.4.0)
from .processor_core import ProcessorCore
from .skip_checker import PatchSkipChecker
from .tile_processor import TileProcessor

# Note: FeatureComputer has been replaced by FeatureOrchestrator in Phase 4.3


# Import from modules (refactored in Phase 3.2)


# Configure logging
logger = logging.getLogger(__name__)

# Processing mode type definition
ProcessingMode = Literal["patches_only", "both", "enriched_only", "reclassify_only"]


class LiDARProcessor:
    """
    Main entry point for processing IGN LiDAR HD data into ML-ready datasets.

    This is the **PUBLIC API** that users interact with directly. It provides high-level
    orchestration for batch processing of LiDAR tiles with classification, feature extraction,
    and dataset generation.

    Architecture Overview:
    =====================

    LiDARProcessor (Public API - This Class)
    │
    ├─→ Configuration Management
    │   ├─ Config validation and optimization
    │   ├─ Hardware detection (GPU/CPU)
    │   └─ Auto-tuning of processing parameters
    │
    ├─→ Batch Orchestration
    │   ├─ Multi-tile processing
    │   ├─ Progress tracking
    │   ├─ Error handling and recovery
    │   └─ Dataset management (train/val/test splits)
    │
    ├─→ TileProcessor (Tile-Level Processing)
    │   ├─ Individual tile loading and validation
    │   ├─ Ground truth integration (WFS, BD TOPO, etc.)
    │   ├─ Feature computation orchestration
    │   ├─ Classification application
    │   ├─ Patch extraction
    │   └─ Output generation (LAZ, NPZ, HDF5, etc.)
    │   │
    │   └─→ ProcessorCore (Low-Level Operations)
    │       ├─ Spatial indexing (KD-tree)
    │       ├─ Neighbor searches
    │       ├─ Point cloud preprocessing
    │       └─ Memory-efficient algorithms
    │
    ├─→ FeatureOrchestrator (Feature Management)
    │   ├─ Feature mode selection (minimal/LOD2/LOD3/full)
    │   ├─ CPU/GPU strategy selection
    │   ├─ Geometric features (normals, curvature, planarity, etc.)
    │   ├─ Multi-scale features (optional)
    │   └─ Spectral features (RGB, NIR, NDVI)
    │
    └─→ Classification Subsystem
        ├─ Classifier (Ground truth integration)
        ├─ Rule-based classification (geometric, spectral)
        ├─ Building detection and refinement
        ├─ Vegetation classification (NDVI-based)
        └─ Transport infrastructure detection

    Key Responsibilities:
    ====================

    1. **User Interface**: Simple, intuitive API for common workflows
       ```python
       processor = LiDARProcessor(config)
       processor.process_tiles()
       ```

    2. **Batch Management**: Process multiple tiles efficiently
       - Parallel processing support (multi-worker)
       - Memory management and cleanup
       - Progress tracking with tqdm
       - Graceful error handling

    3. **Configuration**: Modern config system with smart defaults
       - Preset configurations (asprs_production, lod2_buildings, etc.)
       - Hardware detection and auto-tuning
       - Validation and conflict detection

    4. **Dataset Generation**: Create ML-ready datasets
       - Train/val/test splits
       - Multiple output formats (NPZ, HDF5, LAZ, PyTorch)
       - Metadata tracking
       - Augmentation support

    5. **Delegation**: Route work to specialized components
       - TileProcessor for per-tile operations
       - FeatureOrchestrator for feature computation
       - ClassificationApplier for ground truth

    When to Use:
    ============

    - **Batch processing** of LiDAR tiles from IGN LiDAR HD
    - **Creating ML datasets** for building classification (LOD2/LOD3)
    - **Enriching point clouds** with features and classifications
    - **Production pipelines** with error recovery and monitoring

    When NOT to Use:
    ================

    - Single-tile processing with custom logic → Use TileProcessor directly
    - Feature computation only → Use FeatureOrchestrator directly
    - Custom processing workflows → Use ProcessorCore and other components

    Example Usage:
    =============

    Basic usage with preset:
        >>> from ign_lidar.config import Config
        >>> config = Config.preset('lod2_buildings')
        >>> config.input_dir = '/data/lidar_tiles'
        >>> config.output_dir = '/data/output'
        >>>
        >>> processor = LiDARProcessor(config)
        >>> processor.process_tiles()

    Advanced configuration:
        >>> config = Config(
        ...     input_dir='/data/lidar_tiles',
        ...     output_dir='/data/output',
        ...     mode='lod2',
        ...     use_gpu=True,
        ...     ground_truth={'bd_topo': {'buildings': True}},
        ...     features={'use_rgb': True, 'compute_ndvi': True}
        ... )
        >>> processor = LiDARProcessor(config)
        >>> results = processor.process_tiles()

    Monitoring progress:
        >>> for tile_path, result in processor.process_tiles():
        ...     print(f"Processed {tile_path.name}: {result['num_patches']} patches")

    Related Classes:
    ===============

    - **TileProcessor**: Per-tile orchestration (1 level down)
    - **ProcessorCore**: Low-level operations (2 levels down)
    - **FeatureOrchestrator**: Feature computation management
    - **Classifier**: Ground truth classification
    - **DatasetManager**: ML dataset organization

    Version History:
    ===============

    - v3.2: Single Config class replacing multiple schemas
    - v3.1: Classifier replacing multiple classifier classes
    - v3.0: GPU acceleration with CuPy/cuML support
    - v2.x: Multi-scale features and architectural style detection
    - v1.x: Initial release with LOD2/LOD3 support

    See Also:
    ========

    - Documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
    - Examples: examples/quickstart/
    - Architecture: docs/docs/architecture.md
    """

    def __init__(self, config: Union[DictConfig, Dict] = None, **kwargs):
        """
        Initialize processor with config object or individual parameters (backward compatible).

        Args:
            config: Configuration object (DictConfig or dict) containing all settings.
                   If None, will build config from kwargs for backward compatibility.
            **kwargs: Individual parameters (deprecated, use config instead).
                     Supported for backward compatibility with existing code.

        Config Structure (when using config object):
            processor:
                lod_level: 'LOD2' or 'LOD3'
                processing_mode: 'patches_only', 'both', or 'enriched_only'
                augment: Enable data augmentation
                num_augmentations: Number of augmentations per patch
                bbox: Bounding box (xmin, ymin, xmax, ymax)
                patch_size: Patch size in meters (default: 150.0)
                patch_overlap: Overlap ratio (default: 0.1)
                num_points: Target points per patch (default: 16384)
                use_gpu: GPU acceleration for features (default: False)
                use_gpu_chunked: Chunked GPU processing (default: True)
                gpu_batch_size: GPU batch size (default: 1,000,000)
                preprocess: Apply preprocessing (default: False)
                use_stitching: Enable tile stitching (default: False)
                buffer_size: Buffer zone size in meters (default: 10.0)
                architecture: Target DL architecture (default: 'pointnet++')
                output_format: Output format(s), e.g., 'npz', 'hdf5,laz'

            features:
                include_extra_features: Compute extra building features
                feature_mode: Feature mode ('minimal', 'lod2', 'lod3', 'full')
                k_neighbors: Neighbors for feature computation (None = auto)
                include_architectural_style: Include architectural style
                style_encoding: Style encoding ('constant' or 'multihot')
                use_rgb: Add RGB from orthophotos
                rgb_cache_dir: Cache directory for RGB tiles
                use_infrared: Add NIR from LAZ files
                compute_ndvi: Compute NDVI from RGB+NIR

        Legacy kwargs (deprecated):
            All parameters from previous signature supported for backward compatibility.
            See migration guide for transitioning to config-based approach.
        """
        # Handle config initialization
        if config is None:
            # Build config from kwargs for backward compatibility
            config = self._build_config_from_kwargs(kwargs)
            logger.debug(
                "Built config from legacy kwargs (consider migrating to config-based initialization)"
            )
        elif not isinstance(config, (DictConfig, dict)):
            raise TypeError(
                f"config must be DictConfig or dict, got {type(config).__name__}. "
                f"Pass None and use kwargs for legacy parameter-based initialization."
            )

        # Convert dict to DictConfig if needed
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        # Validate configuration
        self._validate_config(config)

        # Apply auto-optimization if enabled
        if OmegaConf.select(config, "processing.auto_optimize", default=False):
            logger.info(
                "🧠 Auto-optimization enabled - analyzing system capabilities..."
            )

            # Get optimization recommendations
            recommendations = optimization_factory.recommend_optimization(
                OmegaConf.to_object(config)
            )

            if recommendations.get("config_updates"):
                logger.info(
                    f"📈 Applying optimization strategy: {recommendations['strategy'].value}"
                )

                # Apply recommended updates
                for key, value in recommendations["config_updates"].items():
                    if key == "architecture":
                        OmegaConf.update(config, "processing.architecture", value)
                    else:
                        OmegaConf.update(config, f"processor.{key}", value)

                # Log performance expectations
                if recommendations.get("estimated_performance"):
                    logger.info(
                        f"⚡ Expected improvement: {recommendations['estimated_performance']}"
                    )

                # Log any warnings
                for warning in recommendations.get("warnings", []):
                    logger.warning(f"⚠️  {warning}")

        # Store config
        self.config = config

        # Extract commonly used values for convenient access
        self.lod_level = config.processor.lod_level
        self.processing_mode = config.processor.processing_mode
        if self.processing_mode == "enriched_and_patches":
            save_enriched = OmegaConf.select(
                config, "output.save_enriched", default=True
            )
            self.processing_mode = "both" if save_enriched else "patches_only"
        self.patch_size = OmegaConf.select(
            config, "processor.patch_size", default=150.0
        )
        self.num_points = OmegaConf.select(
            config, "processor.num_points", default=16384
        )
        self.architecture = OmegaConf.select(
            config, "processor.architecture", default="pointnet++"
        )
        self.output_format = OmegaConf.select(
            config, "output.format", default="npz"
        )
        self.save_patches = OmegaConf.select(
            config, "output.save_patches", default=False
        )

        # Derive save/only flags from processing mode for internal use
        self.save_enriched_laz = self.processing_mode in ["both", "enriched_only"]
        self.only_enriched_laz = self.processing_mode == "enriched_only"

        logger.info(f"✨ Processing mode: {self.processing_mode}")
        if not self.save_patches:
            logger.info(f"   📦 Patch generation: DISABLED (save_patches=false)")
        logger.info(f"Initialized LiDARProcessor with {self.lod_level}")

        # Validate output format using ConfigValidator
        validated_formats = ConfigValidator.validate_output_format(self.output_format)
        logger.debug(f"Validated output formats: {validated_formats}")

        # Validate processing mode
        ConfigValidator.validate_processing_mode(self.processing_mode)

        # Phase 4.3: Initialize FeatureOrchestrator V5 (consolidated)
        # All optimizations are now built into the main FeatureOrchestrator
        # Phase 2 Session 3: Use FeatureEngine wrapper for cleaner API
        logger.info("🚀 Using FeatureEngine with FeatureOrchestrator V5")
        from .feature_engine import FeatureEngine
        self.feature_engine = FeatureEngine(config)
        
        # Backward compatibility: expose orchestrator directly
        self.feature_orchestrator = self.feature_engine.orchestrator
        self.feature_manager = self.feature_orchestrator  # Backward compatibility alias

        # OPTIMIZATION: Enable GPU profiling if requested (November 2025 Audit - Priority 2 Task 8)
        use_gpu = OmegaConf.select(config, "processor.use_gpu", default=False)
        profile_gpu = OmegaConf.select(config, "processor.profile_gpu", default=False)
        
        # Initialize GPU profiler reference (Priority 2 Task 8: integrate gpu.profiler)
        self.gpu_profiler = None
        
        if use_gpu and profile_gpu:
            try:
                # Enable comprehensive GPU profiling via GPUManager
                from ..core.gpu import GPUManager
                gpu = GPUManager()
                if gpu.gpu_available:
                    self.gpu_profiler = gpu.profiler
                    logger.info("📊 GPU profiling enabled (comprehensive + transfers)")
                    logger.info("   - CUDA event timing")
                    logger.info("   - Memory tracking")
                    logger.info("   - Bottleneck detection")
                    logger.info("   ⚠️  Adds overhead - disable in production")
                else:
                    logger.warning("⚠️  GPU profiling requested but GPU unavailable")
                
                # Also enable transfer profiling for detailed transfer tracking
                from ..optimization.gpu_transfer_profiler import enable_automatic_tracking
                enable_automatic_tracking()
                logger.info("   - Transfer profiling enabled")
            except ImportError as e:
                logger.warning(f"⚠️  GPU profiler not available: {e}")
        elif use_gpu and not profile_gpu:
            logger.debug("GPU profiling disabled (set profile_gpu=true to enable)")

        # Setup stitching configuration and initialize stitcher if needed
        # Handle both old (processor.use_stitching) and new (stitching.enabled) config structures
        if hasattr(config, "stitching") and hasattr(config.stitching, "enabled"):
            use_stitching = config.stitching.enabled
            buffer_size = getattr(config.stitching, "buffer_size", 10.0)
        else:
            use_stitching = config.processor.get("use_stitching", False)
            buffer_size = config.processor.get("buffer_size", 10.0)

        stitching_config = ConfigValidator.setup_stitching_config(
            use_stitching, buffer_size, config.processor.get("stitching_config", None)
        )
        self.stitcher = ConfigValidator.init_stitcher(stitching_config)

        # Phase 2 Session 4: Initialize ClassificationEngine wrapper
        logger.info("🔧 Using ClassificationEngine wrapper")
        from .classification_engine import ClassificationEngine
        self.classification_engine = ClassificationEngine(config, lod_level=self.lod_level)
        
        # Backward compatibility: expose class mapping directly
        self.class_mapping = self.classification_engine.class_mapping
        self.default_class = self.classification_engine.default_class

        # Initialize intelligent skip checker
        self.skip_checker = PatchSkipChecker(
            output_format=self.output_format,
            architecture=self.architecture,
            num_augmentations=config.processor.get("num_augmentations", 3),
            augment=config.processor.get("augment", False),
            validate_content=True,  # Enable content validation
            min_file_size=1024,  # 1KB minimum
            only_enriched_laz=self.only_enriched_laz,
        )

        # Initialize dataset manager for ML dataset creation (with train/val/test splits)
        self.dataset_manager = None
        if config.get("dataset", {}).get("enabled", False):
            dataset_config = DatasetConfig(
                train_ratio=config.dataset.get("train_ratio", 0.7),
                val_ratio=config.dataset.get("val_ratio", 0.15),
                test_ratio=config.dataset.get("test_ratio", 0.15),
                random_seed=config.dataset.get("random_seed", 42),
                split_by_tile=config.dataset.get("split_by_tile", True),
                create_split_dirs=config.dataset.get("create_split_dirs", True),
                patch_sizes=config.dataset.get("patch_sizes", [int(self.patch_size)]),
                balance_across_sizes=config.dataset.get("balance_across_sizes", False),
            )
            # Dataset manager will be initialized in process_directory with output_dir
            self._dataset_config = dataset_config
            logger.info("📊 Dataset mode enabled - will create train/val/test splits")
        else:
            self._dataset_config = None

        # Phase 4.3: Initialize tile processing modules
        self.tile_loader = TileLoader(self.config)

        # Initialize data fetcher for ground truth (BD TOPO, BD Forêt, RPG, Cadastre)
        self.data_fetcher = None
        logger.debug("Checking data sources configuration...")

        if "data_sources" not in config:
            logger.warning(
                "⚠️  'data_sources' not found in config! Ground truth data won't be loaded."
            )

        # Use OmegaConf.select() for compatibility with Hydra configs
        # Support both flat (v4.0) and nested (v5.0) structure
        # Check nested structure FIRST (V5 structure), then fall back to flat structure (V4)
        bd_topo_enabled = OmegaConf.select(
            config, "data_sources.bd_topo.enabled", default=False
        )

        if bd_topo_enabled:
            # Extract from nested structure (V5)
            # Check if features.buildings is a dict (with .enabled) or a boolean (legacy)
            buildings_value = OmegaConf.select(
                config, "data_sources.bd_topo.features.buildings", default=False
            )
            if isinstance(buildings_value, (dict, DictConfig)):
                bd_topo_buildings = buildings_value.get('enabled', False) if hasattr(buildings_value, 'get') else False
            else:
                bd_topo_buildings = bool(buildings_value)
            
            roads_value = OmegaConf.select(
                config, "data_sources.bd_topo.features.roads", default=False
            )
            if isinstance(roads_value, (dict, DictConfig)):
                bd_topo_roads = roads_value.get('enabled', False) if hasattr(roads_value, 'get') else False
            else:
                bd_topo_roads = bool(roads_value)
            
            water_value = OmegaConf.select(
                config, "data_sources.bd_topo.features.water", default=False
            )
            if isinstance(water_value, (dict, DictConfig)):
                bd_topo_water = water_value.get('enabled', False) if hasattr(water_value, 'get') else False
            else:
                bd_topo_water = bool(water_value)
            
            vegetation_value = OmegaConf.select(
                config, "data_sources.bd_topo.features.vegetation", default=False
            )
            if isinstance(vegetation_value, (dict, DictConfig)):
                bd_topo_vegetation = vegetation_value.get('enabled', False) if hasattr(vegetation_value, 'get') else False
            else:
                bd_topo_vegetation = bool(vegetation_value)
            
            # Bridges and power_lines might be in nested OR flat structure
            bridges_value = OmegaConf.select(
                config,
                "data_sources.bd_topo.features.bridges",
                default=OmegaConf.select(
                    config, "data_sources.bd_topo_bridges", default=False
                ),
            )
            if isinstance(bridges_value, (dict, DictConfig)):
                bd_topo_bridges = bridges_value.get('enabled', False) if hasattr(bridges_value, 'get') else False
            else:
                bd_topo_bridges = bool(bridges_value)
            
            power_lines_value = OmegaConf.select(
                config,
                "data_sources.bd_topo.features.power_lines",
                default=OmegaConf.select(
                    config, "data_sources.bd_topo_power_lines", default=False
                ),
            )
            if isinstance(power_lines_value, (dict, DictConfig)):
                bd_topo_power_lines = power_lines_value.get('enabled', False) if hasattr(power_lines_value, 'get') else False
            else:
                bd_topo_power_lines = bool(power_lines_value)

            logger.debug("Extracted BD TOPO features from nested config (V5)")
        else:
            # Fall back to flat structure (V4)
            bd_topo_buildings = OmegaConf.select(
                config, "data_sources.bd_topo_buildings", default=False
            )
            bd_topo_roads = OmegaConf.select(
                config, "data_sources.bd_topo_roads", default=False
            )
            bd_topo_water = OmegaConf.select(
                config, "data_sources.bd_topo_water", default=False
            )
            bd_topo_vegetation = OmegaConf.select(
                config, "data_sources.bd_topo_vegetation", default=False
            )
            bd_topo_bridges = OmegaConf.select(
                config, "data_sources.bd_topo_bridges", default=False
            )
            bd_topo_power_lines = OmegaConf.select(
                config, "data_sources.bd_topo_power_lines", default=False
            )

            # Check if ANY feature is enabled
            bd_topo_enabled = any(
                [
                    bd_topo_buildings,
                    bd_topo_roads,
                    bd_topo_water,
                    bd_topo_vegetation,
                    bd_topo_bridges,
                    bd_topo_power_lines,
                ]
            )

            if bd_topo_enabled:
                logger.debug("Extracted BD TOPO features from flat config (V4)")

        bd_foret_enabled = OmegaConf.select(
            config, "data_sources.bd_foret_enabled", default=False
        )
        rpg_enabled = OmegaConf.select(
            config, "data_sources.rpg_enabled", default=False
        )
        cadastre_enabled = OmegaConf.select(
            config, "data_sources.cadastre_enabled", default=False
        )

        # Log enabled data sources
        logger.info(
            f"   - BD TOPO®: {'✅ Enabled' if bd_topo_enabled else '❌ Disabled'}"
        )
        logger.info(
            f"   - BD Forêt®: {'✅ Enabled' if bd_foret_enabled else '❌ Disabled'}"
        )
        logger.info(f"   - RPG: {'✅ Enabled' if rpg_enabled else '❌ Disabled'}")
        logger.info(
            f"   - Cadastre: {'✅ Enabled' if cadastre_enabled else '❌ Disabled'}"
        )

        if bd_topo_enabled or bd_foret_enabled or rpg_enabled or cadastre_enabled:
            try:
                from ..io.data_fetcher import DataFetchConfig, DataFetcher

                # Build data source configuration
                cache_dir = Path(
                    OmegaConf.select(config, "cache_dir", default="data/cache")
                )

                # Extract BD TOPO features using the flat configuration structure (v4.0)
                # Create configuration with ALL BD TOPO features and parameters
                fetch_config = DataFetchConfig(
                    # BD TOPO® features - use flat structure
                    include_buildings=bd_topo_buildings,
                    include_roads=bd_topo_roads,
                    include_railways=OmegaConf.select(
                        config, "data_sources.bd_topo_railways", default=False
                    ),
                    include_water=bd_topo_water,
                    include_vegetation=bd_topo_vegetation,
                    include_bridges=bd_topo_bridges,
                    include_parking=OmegaConf.select(
                        config, "data_sources.bd_topo_parking", default=False
                    ),
                    include_cemeteries=OmegaConf.select(
                        config, "data_sources.bd_topo_cemeteries", default=False
                    ),
                    include_power_lines=bd_topo_power_lines,
                    include_sports=OmegaConf.select(
                        config, "data_sources.bd_topo_sports", default=False
                    ),
                    # Other data sources
                    include_forest=bd_foret_enabled,
                    include_agriculture=rpg_enabled,
                    include_cadastre=cadastre_enabled,
                    group_by_parcel=OmegaConf.select(
                        config, "data_sources.cadastre_group_by_parcel", default=False
                    ),
                    # Buffer parameters - check both flat and nested structures
                    road_width_fallback=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_road_width_fallback",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.road_width_fallback",
                            default=4.0,
                        ),
                    ),
                    railway_width_fallback=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_railway_width_fallback",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.railway_width_fallback",
                            default=3.5,
                        ),
                    ),
                    power_line_buffer=OmegaConf.select(
                        config,
                        "data_sources.bd_topo_power_line_buffer",
                        default=OmegaConf.select(
                            config,
                            "data_sources.bd_topo.parameters.power_line_buffer",
                            default=2.0,
                        ),
                    ),
                    # RPG year
                    rpg_year=OmegaConf.select(
                        config,
                        "data_sources.rpg_year",
                        default=OmegaConf.select(
                            config, "data_sources.rpg.year", default=2024
                        ),
                    ),
                )

                # Initialize data fetcher
                self.data_fetcher = DataFetcher(
                    cache_dir=cache_dir, config=fetch_config
                )
                logger.info("✅ Data fetcher initialized successfully")
                logger.info(f"   Cache directory: {cache_dir}")
                logger.info("")

                # Log enabled data sources and features
                enabled_sources = []
                if bd_topo_enabled:
                    enabled_features = []
                    if bd_topo_roads:
                        enabled_features.append("roads")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_railways", default=False
                    ):
                        enabled_features.append("railways")
                    if bd_topo_buildings:
                        enabled_features.append("buildings")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_cemeteries", default=False
                    ):
                        enabled_features.append("cemeteries")
                    if bd_topo_power_lines:
                        enabled_features.append("power_lines")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_sports", default=False
                    ):
                        enabled_features.append("sports")
                    if OmegaConf.select(
                        config, "data_sources.bd_topo_parking", default=False
                    ):
                        enabled_features.append("parking")
                    if bd_topo_bridges:
                        enabled_features.append("bridges")
                    if bd_topo_water:
                        enabled_features.append("water")
                    if bd_topo_vegetation:
                        enabled_features.append("vegetation")
                    if enabled_features:
                        enabled_sources.append(
                            f"BD TOPO ({', '.join(enabled_features)})"
                        )
                if bd_foret_enabled:
                    enabled_sources.append("BD Forêt")
                if rpg_enabled:
                    enabled_sources.append("RPG")
                if cadastre_enabled:
                    enabled_sources.append("Cadastre")

                if enabled_sources:
                    logger.info(
                        f"   📦 Enabled data sources: {', '.join(enabled_sources)}"
                    )

            except ImportError as e:
                logger.error(f"❌ Could not initialize data fetcher: {e}")
                logger.error(f"   Ground truth classification will NOT be applied")
                logger.error(
                    f"   Install required packages: pip install geopandas shapely"
                )
                self.data_fetcher = None
        else:
            logger.info(
                f"ℹ️  No data sources enabled - ground truth classification disabled"
            )

        # Phase 2 (v3.5.0): Initialize new manager classes for better separation of concerns
        logger.debug("Initializing I/O and ground truth managers...")
        
        # Initialize TileIOManager for file operations
        from .tile_io_manager import TileIOManager
        input_dir = Path(config.get("input_dir", "."))
        self.tile_io_manager = TileIOManager(input_dir=input_dir)
        logger.debug(f"   ✓ TileIOManager initialized (input_dir: {input_dir})")
        
        # Initialize GroundTruthManager for data prefetching and caching
        from .ground_truth_manager import GroundTruthManager
        cache_dir = Path(OmegaConf.select(config, "cache_dir", default="data/cache"))
        self.ground_truth_manager = GroundTruthManager(
            data_sources_config=config.get("data_sources", {}),
            cache_dir=cache_dir
        )
        logger.debug(f"   ✓ GroundTruthManager initialized (cache_dir: {cache_dir})")

        # Phase 2 Session 5: Initialize TileOrchestrator for tile processing
        # (Must be after data_fetcher initialization to pass it as dependency)
        logger.info("🔧 Initializing TileOrchestrator")
        from .tile_orchestrator import TileOrchestrator
        self.tile_orchestrator = TileOrchestrator(
            config=config,
            feature_orchestrator=self.feature_orchestrator,  # Use the exposed attribute
            classifier=None,  # Will be set later if ground truth is enabled
            reclassifier=None,  # Will be set later if reclassification is enabled
            lod_level=self.lod_level,
            class_mapping=self.class_mapping,
            default_class=self.default_class,
            data_fetcher=self.data_fetcher,  # Pass data_fetcher for DTM augmentation
        )
        logger.debug("   ✓ TileOrchestrator initialized")

        # Phase 4 (v3.9.0): Initialize OptimizationManager for production performance
        self._setup_optimization_manager()

    def _setup_optimization_manager(self):
        """
        Setup Phase 4 optimization manager.
        
        Integrates all Phase 4 optimizations:
        - Phase 4.1: WFS Memory Cache (+10-15%)
        - Phase 4.2: Preprocessing GPU (+10-15%)
        - Phase 4.3: GPU Memory Pooling (+8.5%)
        - Phase 4.4: Batch Multi-Tile Processing (+25-30%)
        - Phase 4.5: Async I/O Pipeline (+12-14%)
        
        Total expected gain: +66-94% (2.66× - 2.94× faster)
        """
        from .optimization_integration import (
            create_optimization_manager,
            ASYNC_IO_AVAILABLE,
            GPU_MEMORY_POOL_AVAILABLE,
            GPU_PROCESSOR_AVAILABLE
        )
        
        logger = logging.getLogger(__name__)
        
        # Check if optimizations are enabled
        enable_optimizations = self.config.get('enable_optimizations', True)
        
        if not enable_optimizations:
            logger.info("Phase 4 optimizations disabled by config")
            self.optimization_manager = None
            return
        
        # Log component availability
        logger.info("🚀 Phase 4 Optimization Components:")
        logger.info(f"   - Async I/O (4.5):        {'✅' if ASYNC_IO_AVAILABLE else '❌'}")
        logger.info(f"   - GPU Memory Pool (4.3):  {'✅' if GPU_MEMORY_POOL_AVAILABLE else '❌'}")
        logger.info(f"   - Batch Processing (4.4): {'✅' if GPU_PROCESSOR_AVAILABLE else '❌'}")
        
        # Extract optimization config with defaults
        enable_async_io = self.config.get('enable_async_io', True)
        async_workers = self.config.get('async_workers', 2)
        tile_cache_size = self.config.get('tile_cache_size', 3)
        enable_batch_processing = self.config.get('enable_batch_processing', True)
        batch_size = self.config.get('batch_size', 4)
        enable_gpu_pooling = self.config.get('enable_gpu_pooling', True)
        gpu_pool_max_size_gb = self.config.get('gpu_pool_max_size_gb', 4.0)
        
        # Create optimization manager
        try:
            self.optimization_manager = create_optimization_manager(
                use_gpu=self.use_gpu,
                enable_all=True,  # Enable all available optimizations
                batch_size=batch_size,
                async_workers=async_workers,
                tile_cache_size=tile_cache_size,
                gpu_pool_max_size_gb=gpu_pool_max_size_gb,
            )
            
            # Initialize with feature orchestrator
            self.optimization_manager.initialize(
                feature_orchestrator=self.feature_orchestrator
            )
            
            logger.info("✅ Phase 4 OptimizationManager initialized")
            logger.info(f"   - Async I/O: {self.optimization_manager.enable_async_io}")
            logger.info(f"   - Batch processing: {self.optimization_manager.enable_batch_processing}")
            logger.info(f"   - GPU pooling: {self.optimization_manager.enable_gpu_pooling}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize OptimizationManager: {e}")
            logger.warning("Falling back to standard processing")
            self.optimization_manager = None

    def __del__(self):
        """Clean up resources on deletion."""
        # Shutdown optimization manager if it exists
        if hasattr(self, 'optimization_manager') and self.optimization_manager:
            try:
                self.optimization_manager.shutdown()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Error shutting down OptimizationManager: {e}")

    def _validate_config(self, config: DictConfig) -> None:
        """Validate configuration object has required fields."""
        required_sections = ["processor", "features"]
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Config missing required section: '{section}'. "
                    f"Available sections: {list(config.keys())}"
                )

        required_processor_fields = ["lod_level", "processing_mode"]
        for field in required_processor_fields:
            if field not in config.processor:
                raise ValueError(
                    f"Config.processor missing required field: '{field}'. "
                    f"Available fields: {list(config.processor.keys())}"
                )

    def _build_config_from_kwargs(self, kwargs: Dict[str, Any]) -> DictConfig:
        """
        Build a config object from legacy kwargs for backward compatibility.

        Args:
            kwargs: Dictionary of legacy parameter names and values

        Returns:
            DictConfig object with processor and features sections
        """
        # Determine processing_mode - old flags take precedence for backward compatibility
        save_enriched = kwargs.get("save_enriched_laz")
        only_enriched = kwargs.get("only_enriched_laz")

        # If old flags are explicitly provided, they override processing_mode
        if save_enriched is not None or only_enriched is not None:
            # Infer from legacy flags (backward compatibility priority)
            save_enriched = save_enriched if save_enriched is not None else False
            only_enriched = only_enriched if only_enriched is not None else False

            if only_enriched:
                processing_mode = "enriched_only"
            elif save_enriched:
                processing_mode = "both"
            else:
                processing_mode = "patches_only"
        else:
            # No old flags provided, use explicit processing_mode or default
            processing_mode = kwargs.get("processing_mode", "patches_only")

        # Create config structure with defaults
        config_dict = {
            "processor": {
                "lod_level": kwargs.get("lod_level", "LOD2"),
                "processing_mode": processing_mode,
                "augment": kwargs.get("augment", False),
                "num_augmentations": kwargs.get("num_augmentations", 3),
                "bbox": kwargs.get("bbox", None),
                "patch_size": kwargs.get("patch_size", 150.0),
                "patch_overlap": kwargs.get("patch_overlap", 0.1),
                "num_points": kwargs.get("num_points", 16384),
                "use_gpu": kwargs.get("use_gpu", False),
                "use_gpu_chunked": kwargs.get("use_gpu_chunked", True),
                "gpu_batch_size": kwargs.get("gpu_batch_size", 1_000_000),
                "preprocess": kwargs.get("preprocess", False),
                "preprocess_config": kwargs.get("preprocess_config", None),
                "use_stitching": kwargs.get("use_stitching", False),
                "buffer_size": kwargs.get("buffer_size", 10.0),
                "stitching_config": kwargs.get("stitching_config", None),
                "architecture": kwargs.get("architecture", "pointnet++"),
                "output_format": kwargs.get("output_format", "npz"),
            },
            "features": {
                "include_extra_features": kwargs.get("include_extra_features", False),
                "feature_mode": kwargs.get("feature_mode", None),
                "k_neighbors": kwargs.get("k_neighbors", None),
                "include_architectural_style": kwargs.get(
                    "include_architectural_style", False
                ),
                "style_encoding": kwargs.get("style_encoding", "constant"),
                "use_rgb": kwargs.get("include_rgb", False),
                "rgb_cache_dir": kwargs.get("rgb_cache_dir", None),
                "use_infrared": kwargs.get("include_infrared", False),
                "compute_ndvi": kwargs.get("compute_ndvi", False),
            },
        }

        return OmegaConf.create(config_dict)

    # Backward compatibility properties
    @property
    def rgb_fetcher(self):
        """Access RGB fetcher (backward compatibility)."""
        return self.feature_engine.rgb_fetcher

    @property
    def infrared_fetcher(self):
        """Access infrared fetcher (backward compatibility)."""
        return self.feature_engine.infrared_fetcher

    @property
    def use_gpu(self):
        """Check if GPU is enabled (backward compatibility)."""
        return self.feature_engine.use_gpu

    @property
    def include_rgb(self):
        """Check if RGB is enabled (backward compatibility)."""
        return self.config.features.use_rgb

    @property
    def include_infrared(self):
        """Check if infrared is enabled (backward compatibility)."""
        return self.config.features.use_infrared

    @property
    def compute_ndvi(self):
        """Check if NDVI computation is enabled (backward compatibility)."""
        return self.config.features.compute_ndvi

    @property
    def include_extra_features(self):
        """Check if extra features are enabled (backward compatibility)."""
        # Support both include_extra and include_extra_features for backward compatibility
        if hasattr(self.config.features, "include_extra"):
            return self.config.features.include_extra
        return self.config.features.get("include_extra_features", False)

    @property
    def k_neighbors(self):
        """Get k neighbors value (backward compatibility)."""
        return self.config.features.k_neighbors

    @property
    def feature_mode(self):
        """Get feature mode (backward compatibility)."""
        return self.config.features.mode

    @property
    def include_architectural_style(self):
        """Check if architectural style is enabled (backward compatibility)."""
        return self.config.features.get("include_architectural_style", False)

    @property
    def style_encoding(self):
        """Get style encoding method (backward compatibility)."""
        return self.config.features.style_encoding

    @property
    def augment(self):
        """Check if augmentation is enabled (backward compatibility)."""
        return self.config.processor.get("augment", False)

    @property
    def num_augmentations(self):
        """Get number of augmentations (backward compatibility)."""
        return self.config.processor.get("num_augmentations", 0)

    @property
    def bbox(self):
        """Get bounding box (backward compatibility)."""
        # bbox is at root level, not in processor section
        return self.config.get("bbox")

    @property
    def patch_overlap(self):
        """Get patch overlap (backward compatibility)."""
        return self.config.processor.get("patch_overlap", 0.0)

    @property
    def use_gpu_chunked(self):
        """Check if chunked GPU processing is enabled (backward compatibility)."""
        return self.config.processor.use_gpu_chunked

    @property
    def gpu_batch_size(self):
        """Get GPU batch size (backward compatibility)."""
        return self.config.processor.gpu_batch_size

    @property
    def preprocess(self):
        """Check if preprocessing is enabled (backward compatibility)."""
        return self.config.processor.preprocess

    @property
    def preprocess_config(self):
        """Get preprocessing config (backward compatibility)."""
        return self.config.processor.preprocess_config

    @property
    def use_stitching(self):
        """Check if stitching is enabled (backward compatibility)."""
        # Handle both old (processor.use_stitching) and new (stitching.enabled) config structures
        if hasattr(self.config, "stitching") and hasattr(
            self.config.stitching, "enabled"
        ):
            return self.config.stitching.enabled
        else:
            return getattr(self.config.processor, "use_stitching", False)

    @property
    def buffer_size(self):
        """Get buffer size (backward compatibility)."""
        # Handle both old (processor.buffer_size) and new (stitching.buffer_size) config structures
        if hasattr(self.config, "stitching") and hasattr(
            self.config.stitching, "buffer_size"
        ):
            return self.config.stitching.buffer_size
        else:
            return getattr(self.config.processor, "buffer_size", 10.0)

    @property
    def rgb_cache_dir(self):
        """Get RGB cache directory (backward compatibility)."""
        return self.config.features.rgb_cache_dir

    def _augment_ground_with_dtm(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment ground points using RGE ALTI DTM.

        ⚡ REFACTORED (v3.5.0 Phase 2 Session 6): Delegates to TileOrchestrator.
        
        This method now acts as a thin wrapper that delegates DTM augmentation
        logic to TileOrchestrator.

        Args:
            points: Original point cloud [N, 3] (X, Y, Z)
            classification: Point classifications [N]
            bbox: Bounding box (minx, miny, maxx, maxy)

        Returns:
            Tuple of (augmented_points, augmented_classification)
        """
        # Delegate to TileOrchestrator
        return self.tile_orchestrator._augment_ground_with_dtm(
            points=points,
            classification=classification,
            bbox=bbox
        )

    def _store_augmentation_stats(self, augmentation_attrs: dict, n_added: int):
        """Store DTM augmentation statistics for later reporting."""
        if not hasattr(self, "_augmentation_stats"):
            self._augmentation_stats = []

        # Extract area distribution
        area_labels = augmentation_attrs.get("augmentation_area", [])
        if len(area_labels) > 0:
            from .classification.dtm_augmentation import AugmentationArea

            # Count points per area
            area_counts = {}
            for area_val in area_labels:
                area_name = (
                    area_val
                    if isinstance(area_val, str)
                    else AugmentationArea(area_val).name
                )
                area_counts[area_name] = area_counts.get(area_name, 0) + 1

            self._augmentation_stats.append(
                {"total_added": n_added, "area_distribution": area_counts}
            )

    def _redownload_tile(self, laz_file: Path) -> bool:
        """
        Attempt to re-download a corrupted tile from IGN WFS.
        
        Delegates to TileIOManager for file operations.

        Args:
            laz_file: Path to the corrupted LAZ file

        Returns:
            True if re-download succeeded, False otherwise
        """
        return self.tile_io_manager.redownload_tile(laz_file)

    def _prefetch_ground_truth_for_tile(self, laz_file: Path) -> Optional[dict]:
        """
        Pre-fetch ground truth data for a single tile.
        
        Delegates to GroundTruthManager for prefetching and caching.

        Args:
            laz_file: Path to LAZ file to prefetch data for

        Returns:
            Dictionary containing fetched ground truth data, or None if failed
        """
        return self.ground_truth_manager.prefetch_ground_truth_for_tile(laz_file)

    def _prefetch_ground_truth(self, laz_files: List[Path]):
        """
        Pre-fetch ground truth data for all tiles to warm up the cache.
        
        Delegates to GroundTruthManager for batch prefetching.

        Args:
            laz_files: List of LAZ files to prefetch data for
        """
        results = self.ground_truth_manager.prefetch_ground_truth_batch(
            laz_files, show_progress=True
        )
        logger.info(
            f"✅ Ground truth pre-fetched ({len(results)}/{len(laz_files)} tiles cached)"
        )
        if success_count < len(laz_files):
            logger.warning(
                f"⚠️  {len(laz_files) - success_count} tiles failed to prefetch (will retry during processing)"
            )

    def process_tile_v2(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Process a single LAZ tile using refactored TileProcessor (v3.4.0).

        This is the NEW facade-style method that delegates to TileProcessor.
        It replaces the monolithic _process_tile_core() with clean component orchestration.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist

        Returns:
            Number of patches created (0 if skipped)

        Note:
            This method uses the new component architecture:
            - ProcessorCore: Configuration & initialization
            - TileProcessor: Orchestrates all processing steps
            - ClassificationApplier: Ground truth & classification
            - PatchExtractor: Patch extraction
            - OutputWriter: Multi-format output

        See Also:
            process_tile: Original method (backward compatible)
            TileProcessor: Core processing coordinator
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""

        # Skip check (metadata-based + output-based)
        if skip_existing:
            # Check metadata
            metadata_mgr = ProcessingMetadata(output_dir)
            should_reprocess, reprocess_reason = metadata_mgr.should_reprocess(
                laz_file.stem, self.config
            )

            if not should_reprocess:
                logger.info(
                    f"{progress_prefix} ⏭️  {laz_file.name}: "
                    f"Already processed with same config, skipping"
                )
                return 0

            # Check outputs if no metadata
            if reprocess_reason == "no_metadata_found":
                should_skip, skip_info = self.skip_checker.should_skip_tile(
                    tile_path=laz_file,
                    output_dir=output_dir,
                    expected_patches=None,
                )

                if should_skip:
                    logger.info(f"{progress_prefix} ⏭️  {laz_file.name}: {skip_info}")
                    return 0

        # Initialize TileProcessor (lazy initialization for backward compatibility)
        if not hasattr(self, "_tile_processor"):
            logger.info("🚀 Initializing TileProcessor with refactored components...")

            # Create component instances
            patch_extractor = PatchExtractor(self.config)
            classification_applier = ClassificationApplier(
                self.config, self.data_fetcher
            )
            output_writer = OutputWriter(self.config, self.dataset_manager)

            # Create TileProcessor coordinator
            self._tile_processor = TileProcessor(
                config=self.config,
                feature_orchestrator=self.feature_orchestrator,
                patch_extractor=patch_extractor,
                classification_applier=classification_applier,
                output_writer=output_writer,
                tile_loader=self.tile_loader,
            )

            logger.info("✅ TileProcessor initialized successfully")

        # Delegate to TileProcessor
        num_patches = self._tile_processor.process_tile(
            laz_file=laz_file,
            output_dir=output_dir,
            tile_data=None,  # TileProcessor will load if needed
            # Feature request: Pass prefetched data (see OPTIMIZATION.md)
            prefetched_ground_truth=None,
            progress_prefix=progress_prefix,
            # Feature request: Extract from dataset_manager (see OPTIMIZATION.md)
            tile_split=None,
        )

        return num_patches

    def _process_tile_with_data(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_data: dict,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
        prefetched_ground_truth: dict = None,
    ) -> int:
        """
        Process a single LAZ tile with pre-loaded data and optionally pre-fetched ground truth.

        OPTIMIZATION: Phase 2 - Pipeline Optimization
        This method accepts pre-loaded tile data and optionally pre-fetched ground truth,
        allowing both I/O and ground truth fetching to be pipelined for better GPU utilization.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_data: Pre-loaded tile data from tile_loader
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist
            prefetched_ground_truth: Pre-fetched ground truth data (optional)

        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""

        # If tile data loading failed, log and skip
        if tile_data is None:
            logger.error(f"{progress_prefix} ✗ Failed to load tile: {laz_file.name}")
            return 0

        # Check if we should skip (but skip the tile loading since it's already done)
        if skip_existing:
            # Quick check for existing outputs
            should_skip, skip_info = self.skip_checker.should_skip_tile(
                tile_path=laz_file,
                output_dir=output_dir,
                expected_patches=None,
                save_enriched=self.save_enriched_laz,
                include_rgb=self.config.features.use_rgb,
                include_infrared=self.config.features.use_infrared,
                compute_ndvi=self.config.features.compute_ndvi,
                include_extra_features=self.include_extra_features,
                include_classification=OmegaConf.select(
                    self.config, "data_sources.bd_topo.enabled", default=False
                ),
                include_forest=OmegaConf.select(
                    self.config, "data_sources.bd_foret.enabled", default=False
                ),
                include_agriculture=OmegaConf.select(
                    self.config, "data_sources.rpg.enabled", default=False
                ),
                include_cadastre=OmegaConf.select(
                    self.config, "data_sources.cadastre.enabled", default=False
                ),
            )

            if should_skip:
                skip_msg = self.skip_checker.format_skip_message(laz_file, skip_info)
                logger.info(f"{progress_prefix} {skip_msg}")
                return 0

        # Continue with normal tile processing (tile_data already loaded)
        logger.info(f"{progress_prefix} Processing: {laz_file.name}")

        # If we have prefetched ground truth, the cache should be warm
        # This eliminates the network I/O delay during the actual processing
        if prefetched_ground_truth is not None:
            logger.debug(
                f"{progress_prefix} ✅ Ground truth cache warmed via prefetching"
            )

        # ⚡ OPTIMIZATION: Reuse pre-loaded tile_data instead of calling process_tile
        # This eliminates redundant LAZ file loading (saves ~2-3 seconds per tile)
        # Pass tile_data directly to _process_tile_core
        return self._process_tile_core(
            laz_file, output_dir, tile_data, tile_idx, total_tiles, skip_existing
        )

    def process_tile(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Process a single LAZ tile.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist

        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""

        # Use intelligent skip checker to validate existing outputs
        if skip_existing:
            # First check metadata-based skip (config changes)
            metadata_mgr = ProcessingMetadata(output_dir)
            should_reprocess, reprocess_reason = metadata_mgr.should_reprocess(
                laz_file.stem, self.config
            )

            if should_reprocess:
                if reprocess_reason == "config_changed":
                    logger.info(
                        f"{progress_prefix} Reprocessing {laz_file.name}: "
                        f"Configuration changed since last processing"
                    )
                elif reprocess_reason == "no_metadata_found":
                    logger.debug(
                        f"{progress_prefix} No metadata found for {laz_file.name}, "
                        f"will check for existing outputs"
                    )
                elif reprocess_reason and "output_file_missing" in reprocess_reason:
                    logger.info(
                        f"{progress_prefix} Reprocessing {laz_file.name}: "
                        f"Output file missing"
                    )
            else:
                # Metadata indicates we can skip (config unchanged and outputs exist)
                logger.info(
                    f"{progress_prefix} ⏭️  {laz_file.name}: "
                    f"Already processed with same config, skipping"
                )
                return 0

            # If no metadata or reprocessing needed, use output-based skip checker
            if reprocess_reason == "no_metadata_found":
                should_skip, skip_info = self.skip_checker.should_skip_tile(
                    tile_path=laz_file,
                    output_dir=output_dir,
                    expected_patches=None,  # We don't know expected count beforehand
                    save_enriched=self.save_enriched_laz,
                    include_rgb=self.config.features.use_rgb,
                    include_infrared=self.config.features.use_infrared,
                    compute_ndvi=self.config.features.compute_ndvi,
                    include_extra_features=self.include_extra_features,
                    include_classification=OmegaConf.select(
                        self.config, "data_sources.bd_topo.enabled", default=False
                    )
                    and OmegaConf.select(
                        self.config,
                        "data_sources.bd_topo.features.buildings",
                        default=False,
                    ),
                    include_forest=OmegaConf.select(
                        self.config, "data_sources.bd_foret.enabled", default=False
                    ),
                    include_agriculture=OmegaConf.select(
                        self.config, "data_sources.rpg.enabled", default=False
                    ),
                    include_cadastre=OmegaConf.select(
                        self.config, "data_sources.cadastre.enabled", default=False
                    ),
                )

                if should_skip:
                    # Format detailed skip message
                    skip_msg = self.skip_checker.format_skip_message(
                        laz_file, skip_info
                    )
                    logger.info(f"{progress_prefix} {skip_msg}")
                    return 0
                else:
                    # Log reason for processing (helpful for debugging)
                    reason = skip_info.get("reason", "unknown")
                    if reason == "no_patches_found":
                        logger.debug(
                            f"{progress_prefix} Processing: No existing outputs found"
                        )
                    elif reason == "enriched_laz_invalid":
                        missing_features = skip_info.get("missing_features", [])
                        logger.info(
                            f"{progress_prefix} Reprocessing: Enriched LAZ missing features "
                            f"({', '.join(missing_features[:3])}...)"
                        )
                    elif reason == "corrupted_patches_found":
                        num_corrupted = skip_info.get("corrupted_count", 0)
                        logger.info(
                            f"{progress_prefix} Reprocessing: Found {num_corrupted} "
                            f"corrupted patches"
                        )

        logger.info(f"{progress_prefix} Processing: {laz_file.name}")

        # 1. Load tile data using TileLoader module (Phase 3.4)
        tile_data = self.tile_loader.load_tile(laz_file, max_retries=2)

        if tile_data is None:
            logger.error(f"  ✗ Failed to load tile: {laz_file.name}")
            return 0

        # Validate tile has sufficient points
        if not self.tile_loader.validate_tile(tile_data):
            logger.warning(f"  ⚠️  Insufficient points in tile: {laz_file.name}")
            return 0

        # Delegate to core processing method
        return self._process_tile_core(
            laz_file, output_dir, tile_data, tile_idx, total_tiles, skip_existing
        )

    def _process_tile_core(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_data: dict,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True,
    ) -> int:
        """
        Core tile processing logic that works with pre-loaded tile data.

        ⚡ REFACTORED (v3.5.0 Phase 2 Session 5): Delegates to TileOrchestrator.
        
        This method now acts as a thin wrapper that delegates the complex tile processing
        logic to TileOrchestrator, significantly reducing the size of LiDARProcessor.

        Args:
            laz_file: Path to LAZ file (for output naming)
            output_dir: Output directory
            tile_data: Pre-loaded tile data from TileLoader
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist

        Returns:
            Number of patches created (0 if skipped)
        """
        # Delegate to TileOrchestrator for all tile processing logic
        return self.tile_orchestrator.process_tile_core(
            laz_file=laz_file,
            output_dir=output_dir,
            tile_data=tile_data,
            tile_idx=tile_idx,
            total_tiles=total_tiles,
            skip_existing=skip_existing,
        )

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
        save_metadata: bool = True,
        skip_existing: bool = True,
    ) -> int:
        """
        Process directory of LAZ files.

        Args:
            input_dir: Directory containing LAZ files
            output_dir: Output directory
            num_workers: Number of parallel workers
            save_metadata: Whether to save stats.json
            skip_existing: Skip tiles that already have patches in output

        Returns:
            Total number of patches created
        """
        # Phase 4: Route to optimized processing if OptimizationManager is available
        if hasattr(self, 'optimization_manager') and self.optimization_manager is not None:
            logger.info("🚀 Using Phase 4 optimized processing")
            return self._process_directory_optimized(
                input_dir=input_dir,
                output_dir=output_dir,
                save_metadata=save_metadata,
                skip_existing=skip_existing,
            )
        
        # Otherwise, use standard sequential processing
        logger.info("Using standard sequential processing")
        return self._process_directory_sequential(
            input_dir=input_dir,
            output_dir=output_dir,
            num_workers=num_workers,
            save_metadata=save_metadata,
            skip_existing=skip_existing,
        )

    def _process_directory_sequential(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
        save_metadata: bool = True,
        skip_existing: bool = True,
    ) -> int:
        """
        Process directory sequentially (legacy method).
        
        This is the original implementation kept for backward compatibility
        and as fallback when Phase 4 optimizations are unavailable.
        """
        start_time = time.time()

        # Initialize dataset manager if enabled
        if self._dataset_config is not None:
            self.dataset_manager = DatasetManager(
                output_dir=output_dir,
                config=self._dataset_config,
                patch_size=int(self.patch_size),
            )
            logger.info(f"📊 Dataset manager initialized for {output_dir}")

        # Check system memory and adjust workers if needed
        try:
            import psutil

            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            available_gb = mem.available / (1024**3)
            swap_percent = swap.percent

            logger.info(f"System Memory: {available_gb:.1f}GB available")

            # If swap is heavily used, reduce workers automatically
            if swap_percent > 50:
                logger.warning(f"⚠️  High swap usage detected ({swap_percent:.0f}%)")
                logger.warning("⚠️  Memory pressure detected - reducing workers to 1")
                num_workers = 1

            # Processing needs ~2-3GB per worker
            min_gb_per_worker = 2.5
            max_safe_workers = int(available_gb / min_gb_per_worker)

            if num_workers > max_safe_workers:
                logger.warning(f"⚠️  Limited RAM ({available_gb:.1f}GB available)")
                logger.warning(
                    f"⚠️  Reducing workers from {num_workers} "
                    f"to {max(1, max_safe_workers)}"
                )
                num_workers = max(1, max_safe_workers)

        except ImportError:
            logger.debug("psutil not available - skipping memory checks")

        # Find LAZ files (recursively) - exclude enriched files to avoid reprocessing
        laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))

        # Filter out enriched files (those ending with _enriched.laz)
        laz_files = [f for f in laz_files if not f.stem.endswith("_enriched")]

        if not laz_files:
            logger.error(f"No LAZ files found in {input_dir}")
            return 0

        total_tiles = len(laz_files)
        logger.info(f"Found {total_tiles} LAZ files")

        # OPTIMIZATION: Use tile-by-tile prefetching instead of bulk prefetching
        # This reduces memory usage and allows processing to start immediately
        k_display = self.k_neighbors or "auto"
        logger.info(
            f"Configuration: LOD={self.lod_level} | k={k_display} | "
            f"patch_size={self.patch_size}m | augment={self.augment}"
        )
        logger.info("")

        # Initialize metadata manager
        metadata_mgr = MetadataManager(output_dir) if save_metadata else None

        # Copy directory structure from source
        if metadata_mgr:
            logger.info("Copying directory structure from source...")
            metadata_mgr.copy_directory_structure(input_dir)

        # Process files
        tiles_processed = 0
        tiles_skipped = 0

        if num_workers > 1:
            # Check for GPU + multiprocessing conflict
            if self.feature_orchestrator.use_gpu:
                logger.warning(
                    "⚠️  GPU acceleration is not compatible with multiprocessing due to CUDA context limitations"
                )
                logger.warning(
                    "🔧 Automatically disabling GPU for multiprocessing mode"
                )
                logger.warning(
                    "💡 To use GPU: set num_workers=1, or disable GPU: use_gpu=false"
                )

                # Completely disable GPU for multiprocessing to prevent CUDA context issues
                disable_gpu_for_multiprocessing()

                # Disable GPU in feature orchestrator configuration
                self.feature_orchestrator.use_gpu = False
                if hasattr(self.feature_orchestrator, "computer"):
                    self.feature_orchestrator.computer.use_gpu = False
                if hasattr(self.feature_orchestrator, "config"):
                    if hasattr(self.feature_orchestrator.config, "processor"):
                        self.feature_orchestrator.config.processor.use_gpu = False
                    if hasattr(self.feature_orchestrator.config, "features"):
                        if hasattr(
                            self.feature_orchestrator.config.features,
                            "gpu_optimization",
                        ):
                            self.feature_orchestrator.config.features.gpu_optimization.enabled = (
                                False
                            )

            logger.info(f"🚀 Processing with {num_workers} parallel workers")
            logger.info("=" * 70)

            # For parallel processing, we can't easily pass tile index
            process_func = partial(
                self.process_tile,
                output_dir=output_dir,
                total_tiles=total_tiles,
                skip_existing=skip_existing,
            )

            with mp.Pool(num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_func, laz_files),
                        total=total_tiles,
                        desc="Processing tiles",
                        unit="tile",
                    )
                )

            # Handle both dict and int return types for backwards compatibility
            num_patches_list = []
            for r in results:
                if isinstance(r, dict):
                    num_patches_list.append(r.get("num_patches", 0))
                else:
                    num_patches_list.append(r)

            total_patches = sum(num_patches_list)
            tiles_skipped = sum(1 for n in num_patches_list if n == 0)
            tiles_processed = total_tiles - tiles_skipped
        else:
            # ✅ OPTIMIZATION: Sequential processing with tile-by-tile prefetching (Phase 2)
            # Double-buffering: Load tile N+1 and prefetch ground truth N+1 while GPU processes tile N
            # Expected speedup: +40-80% (eliminates both I/O stalls and ground truth fetch delays)
            logger.info("🔄 Processing sequentially with tile-by-tile prefetching")
            logger.info("=" * 70)

            total_patches = 0

            # Use ThreadPoolExecutor for async I/O and ground truth prefetching
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=3
            ) as io_pool:  # 3 workers: tile loading + ground truth fetching + spare
                # Prefetch first tile and its ground truth
                if laz_files:
                    next_tile_future = io_pool.submit(
                        self.tile_loader.load_tile, laz_files[0], max_retries=2
                    )
                    # Prefetch ground truth for first tile if data fetcher is available
                    if self.data_fetcher is not None:
                        next_ground_truth_future = io_pool.submit(
                            self._prefetch_ground_truth_for_tile, laz_files[0]
                        )
                    else:
                        next_ground_truth_future = None
                else:
                    next_tile_future = None
                    next_ground_truth_future = None

                for idx, laz_file in enumerate(laz_files, 1):
                    # Wait for prefetched tile data and ground truth
                    tile_data = (
                        next_tile_future.result()
                        if next_tile_future is not None
                        else None
                    )
                    prefetched_ground_truth = (
                        next_ground_truth_future.result()
                        if next_ground_truth_future is not None
                        else None
                    )

                    # Start prefetching NEXT tile and ground truth (async I/O, parallel with GPU processing)
                    if idx < len(laz_files):
                        next_tile_future = io_pool.submit(
                            self.tile_loader.load_tile, laz_files[idx], max_retries=2
                        )
                        # Prefetch ground truth for next tile if data fetcher is available
                        if self.data_fetcher is not None:
                            next_ground_truth_future = io_pool.submit(
                                self._prefetch_ground_truth_for_tile, laz_files[idx]
                            )
                        else:
                            next_ground_truth_future = None
                    else:
                        next_tile_future = None
                        next_ground_truth_future = None

                    # Process current tile (GPU busy, I/O and ground truth fetching run in parallel)
                    result = self._process_tile_with_data(
                        laz_file,
                        output_dir,
                        tile_data,
                        tile_idx=idx,
                        total_tiles=total_tiles,
                        skip_existing=skip_existing,
                        prefetched_ground_truth=prefetched_ground_truth,
                    )

                    # Handle both dict and int return types for backwards compatibility
                    if isinstance(result, dict):
                        num_patches = result.get("num_patches", 0)
                    else:
                        num_patches = result

                    total_patches += num_patches

                    if num_patches == 0:
                        tiles_skipped += 1
                    else:
                        tiles_processed += 1

                # ✅ OPTIMIZATION: Smart garbage collection (Phase 1 - Quick Win)
                # Fast cleanup after each tile (generation 0 only - ~5ms)
                del result
                if "tile_data" in locals():
                    del tile_data
                if "all_features" in locals():
                    del all_features
                gc.collect(generation=0)

                # Deep cleanup every 10 tiles (GPU + full GC)
                if idx % 10 == 0:
                    # GPU memory cleanup (if using CuPy)
                    try:
                        import cupy as cp

                        mempool = cp.get_default_memory_pool()
                        pinned_mempool = cp.get_default_pinned_memory_pool()
                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
                        logger.debug("  🎮 GPU memory freed")
                    except:
                        pass

                    # Full CPU garbage collection
                    gc.collect()

                    # Log memory status
                    try:
                        import psutil

                        mem = psutil.virtual_memory()
                        logger.debug(
                            f"  💾 System Memory: {mem.available/(1024**3):.1f}GB available"
                        )
                        try:
                            import cupy as cp

                            free_mem, total_mem = cp.cuda.Device().mem_info
                            logger.debug(
                                f"  🎮 GPU Memory: {free_mem/(1024**3):.1f}GB / {total_mem/(1024**3):.1f}GB free"
                            )
                        except:
                            pass
                    except ImportError:
                        pass

        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Processing Summary:")
        logger.info(f"  Total tiles: {total_tiles}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  📦 Total patches created: {total_patches}")
        logger.info("=" * 70)

        # Save metadata
        if metadata_mgr:
            processing_time = time.time() - start_time
            stats = metadata_mgr.create_processing_stats(
                input_dir=input_dir,
                num_tiles=len(laz_files),
                num_patches=total_patches,
                lod_level=self.lod_level,
                k_neighbors=self.k_neighbors,
                patch_size=self.patch_size,
                augmentation=self.augment,
                num_augmentations=self.num_augmentations,
            )
            stats["processing_time_seconds"] = round(processing_time, 2)
            metadata_mgr.save_stats(stats)

        # Priority 2 Task 8: Print GPU profiling report if enabled
        if self.gpu_profiler is not None:
            try:
                logger.info("")
                logger.info("=" * 70)
                logger.info("📊 GPU Profiling Report:")
                logger.info("=" * 70)
                self.gpu_profiler.print_report()
                logger.info("=" * 70)
            except Exception as e:
                logger.warning(f"⚠️  Failed to print GPU profiling report: {e}")

        # Save dataset metadata if dataset manager is enabled
        if self.dataset_manager is not None:
            processing_time = time.time() - start_time
            additional_info = {
                "lod_level": self.lod_level,
                "architecture": self.architecture,
                "patch_size_meters": self.patch_size,
                "num_points": self.num_points,
                "augmentation_enabled": self.config.processor.get("augment", False),
                "num_augmentations": self.config.processor.get("num_augmentations", 0),
                "processing_time_seconds": round(processing_time, 2),
                "tiles_processed": tiles_processed,
                "tiles_skipped": tiles_skipped,
            }
            self.dataset_manager.save_metadata(additional_info=additional_info)

        # Finalize stateful writers (PTv3 dataset: flushes meta.json + split
        # counts). Without this, coord/feat/segment.npy are written but the
        # dataset has no meta.json and Pointcept can't sanity-check it.
        _orch = getattr(self, "tile_orchestrator", None)
        _ow = getattr(_orch, "_output_writer", None) if _orch is not None else None
        if _ow is not None:
            try:
                summary = _ow.finalize()
                logger.info(f"  ✅ PTv3 dataset finalized — splits {summary}")
            except Exception as e:
                logger.warning(f"⚠️  OutputWriter.finalize() failed: {e}")

        return total_patches

    def _process_directory_optimized(
        self,
        input_dir: Path,
        output_dir: Path,
        save_metadata: bool = True,
        skip_existing: bool = True,
    ) -> int:
        """
        Process directory using Phase 4 optimizations.
        
        Utilizes:
        - Async I/O for tile loading
        - Batch processing on GPU
        - GPU memory pooling
        - WFS memory cache
        
        Expected performance: +66-94% faster than sequential
        """
        start_time = time.time()
        
        # Initialize dataset manager if enabled
        if self._dataset_config is not None:
            self.dataset_manager = DatasetManager(
                output_dir=output_dir,
                config=self._dataset_config,
                patch_size=int(self.patch_size),
            )
            logger.info(f"📊 Dataset manager initialized for {output_dir}")
        
        # Find LAZ files - exclude enriched files
        laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))
        laz_files = [f for f in laz_files if not f.stem.endswith("_enriched")]
        
        if not laz_files:
            logger.error(f"No LAZ files found in {input_dir}")
            return 0
        
        total_tiles = len(laz_files)
        logger.info(f"Found {total_tiles} LAZ files")
        
        k_display = self.k_neighbors or "auto"
        logger.info(
            f"Configuration: LOD={self.lod_level} | k={k_display} | "
            f"patch_size={self.patch_size}m | augment={self.augment}"
        )
        logger.info("")
        
        # Initialize metadata manager
        metadata_mgr = MetadataManager(output_dir) if save_metadata else None
        
        if metadata_mgr:
            logger.info("Copying directory structure from source...")
            metadata_mgr.copy_directory_structure(input_dir)
        
        # Define processing function for a single tile
        def process_tile_optimized(tile_data, ground_truth):
            """Process a single tile with all Phase 4 optimizations."""
            try:
                # Extract tile path from tile_data
                tile_path = Path(tile_data.file_path) if hasattr(tile_data, 'file_path') else None
                
                # Check if should skip
                # NOTE: should_skip_tile() returns a (bool, dict) tuple — it must
                # be unpacked. Testing the tuple directly is always truthy and
                # would skip every tile (0 patches ever created).
                if skip_existing and tile_path:
                    should_skip, skip_info = self.skip_checker.should_skip_tile(
                        tile_path, output_dir
                    )
                    if should_skip:
                        logger.debug(
                            f"⏭️  Skipping {tile_path.name} "
                            f"({skip_info.get('reason', 'patches exist')})"
                        )
                        return {'num_patches': 0, 'num_points': len(tile_data.points), 'skipped': True}
                
                # Process using tile orchestrator
                result = self.tile_orchestrator.process_tile(
                    tile_data=tile_data,
                    output_dir=output_dir,
                    ground_truth=ground_truth,
                    skip_checker=self.skip_checker,
                    dataset_manager=self.dataset_manager,
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing tile: {e}")
                return {'num_patches': 0, 'num_points': 0, 'error': str(e)}
        
        # Process tiles with optimizations
        logger.info("🚀 Processing with Phase 4 optimizations...")
        
        gt_enabled = bool(OmegaConf.select(
            self.config, "ground_truth.enabled", default=False
        ))
        results = self.optimization_manager.process_tiles_optimized(
            tile_paths=laz_files,
            processor_func=process_tile_optimized,
            fetch_ground_truth=gt_enabled,
        )
        
        # Aggregate results
        total_patches = sum(r.get('num_patches', 0) for r in results if r)
        tiles_processed = sum(1 for r in results if r and not r.get('skipped', False))
        tiles_skipped = sum(1 for r in results if r and r.get('skipped', False))
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Processing Summary:")
        logger.info(f"  Total tiles: {total_tiles}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  📦 Total patches created: {total_patches}")
        logger.info("=" * 70)
        
        # Print optimization statistics
        print_stats = self.config.get('print_optimization_stats', True)
        if print_stats:
            self.optimization_manager.print_stats()
        
        # Save metadata
        if metadata_mgr:
            processing_time = time.time() - start_time
            stats = metadata_mgr.create_processing_stats(
                input_dir=input_dir,
                num_tiles=len(laz_files),
                num_patches=total_patches,
                lod_level=self.lod_level,
                k_neighbors=self.k_neighbors,
                patch_size=self.patch_size,
                augmentation=self.augment,
                num_augmentations=self.num_augmentations,
            )
            stats["processing_time_seconds"] = round(processing_time, 2)
            
            # Add Phase 4 optimization stats
            opt_stats = self.optimization_manager.get_stats()
            stats["phase4_optimizations"] = opt_stats
            
            metadata_mgr.save_stats(stats)
        
        # Priority 2 Task 8: Print GPU profiling report if enabled
        if self.gpu_profiler is not None:
            try:
                logger.info("")
                logger.info("=" * 70)
                logger.info("📊 GPU Profiling Report:")
                logger.info("=" * 70)
                self.gpu_profiler.print_report()
                logger.info("=" * 70)
            except Exception as e:
                logger.warning(f"⚠️  Failed to print GPU profiling report: {e}")
        
        # Save dataset metadata if dataset manager is enabled
        if self.dataset_manager is not None:
            processing_time = time.time() - start_time
            opt_stats = self.optimization_manager.get_stats()
            
            additional_info = {
                "lod_level": self.lod_level,
                "architecture": self.architecture,
                "patch_size_meters": self.patch_size,
                "num_points": self.num_points,
                "augmentation_enabled": self.config.processor.get("augment", False),
                "num_augmentations": self.config.processor.get("num_augmentations", 0),
                "processing_time_seconds": round(processing_time, 2),
                "tiles_processed": tiles_processed,
                "tiles_skipped": tiles_skipped,
                "phase4_optimizations": opt_stats,
            }
            self.dataset_manager.save_metadata(additional_info=additional_info)
        
        return total_patches

    def reclassify_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        chunk_size: int = 100000,
        show_progress: bool = True,
        skip_existing: bool = True,
        acceleration_mode: str = "auto",
    ) -> int:
        """
        Reclassify all LAZ files in a directory with BD TOPO® ground truth.

        This is an optimized mode that only updates classification codes using
        spatial indexing for fast processing. Use this when you already have
        enriched tiles but need to apply/update ground truth classification.

        Acceleration modes:
        - 'cpu': Use CPU with STRtree spatial indexing (~5-10 min for 18M points)
        - 'gpu': Use RAPIDS cuSpatial GPU acceleration (~1-2 min for 18M points)
        - 'gpu+cuml': Use full RAPIDS stack (~30-60 sec for 18M points)
        - 'auto': Automatically select best available backend (recommended)

        Args:
            input_dir: Directory containing enriched LAZ files
            output_dir: Output directory for reclassified files
            cache_dir: Cache directory for ground truth data (default: data/cache)
            chunk_size: Points per processing chunk (default: 100,000)
            show_progress: Show progress bars (default: True)
            skip_existing: Skip already reclassified files (default: True)
            acceleration_mode: Acceleration backend ('cpu', 'gpu', 'gpu+cuml', 'auto')

        Returns:
            Number of files processed
        """
        from ..io.data_fetcher import DataFetchConfig, DataFetcher

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if cache_dir is None:
            cache_dir = Path("data/cache")

        logger.info("=" * 80)
        logger.info("🔄 RECLASSIFICATION MODE - Optimized Ground Truth Application")
        logger.info("=" * 80)
        logger.info(f"Input:  {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Cache:  {cache_dir}")
        logger.info(f"Chunk size: {chunk_size:,} points")
        logger.info("=" * 80)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all LAZ files - exclude enriched files to avoid reprocessing
        laz_files = sorted(input_dir.glob("*.laz"))
        laz_files = [f for f in laz_files if not f.stem.endswith("_enriched")]

        if not laz_files:
            logger.warning(f"No LAZ files found in {input_dir}")
            return 0

        logger.info(f"Found {len(laz_files)} LAZ files to process")

        # Initialize data fetcher with all BD TOPO features
        logger.info("\n📍 Initializing BD TOPO® data fetcher...")

        config = DataFetchConfig(
            include_buildings=True,
            include_roads=True,
            include_railways=True,
            include_water=True,
            include_vegetation=True,
            include_bridges=True,
            include_parking=True,
            include_cemeteries=True,
            include_power_lines=True,
            include_sports=True,
            include_forest=False,
            include_agriculture=False,
            include_cadastre=False,
            road_width_fallback=4.0,
            railway_width_fallback=3.5,
            power_line_buffer=2.0,
        )

        data_fetcher = DataFetcher(cache_dir=cache_dir, config=config)

        # Initialize reclassifier with acceleration mode
        reclassifier = Reclassifier(
            chunk_size=chunk_size,
            show_progress=show_progress,
            acceleration_mode=acceleration_mode,
        )

        # Process each file
        tiles_processed = 0
        tiles_skipped = 0
        start_time = time.time()

        for idx, laz_file in enumerate(laz_files, 1):
            logger.info(f"\n[{idx}/{len(laz_files)}] Processing: {laz_file.name}")

            # Output file path
            output_file = output_dir / f"{laz_file.stem}_reclassified.laz"

            # Skip if already exists
            if skip_existing and output_file.exists():
                logger.info(f"  ⏭️  Skipped (already exists): {output_file.name}")
                tiles_skipped += 1
                continue

            try:
                # 1. Load file to get bbox
                las = laspy.read(str(laz_file))
                points = np.vstack([las.x, las.y, las.z]).T

                bbox = (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )

                logger.info(f"  📦 Points: {len(points):,}")
                logger.info(f"  📏 Bbox: {bbox}")

                # 2. Fetch ground truth for this tile's bbox
                logger.info(f"  🗺️  Fetching ground truth...")
                gt_data = data_fetcher.fetch_all(bbox=bbox, use_cache=True)

                if not gt_data or "ground_truth" not in gt_data:
                    logger.warning(f"  ⚠️  No ground truth data available, skipping")
                    tiles_skipped += 1
                    continue

                ground_truth_features = gt_data["ground_truth"]

                # Log what was fetched
                n_features = sum(
                    1
                    for gdf in ground_truth_features.values()
                    if gdf is not None and len(gdf) > 0
                )
                logger.info(f"  ✓ Fetched {n_features} feature types")

                # 3. Reclassify
                logger.info(f"  🎯 Reclassifying...")
                stats = reclassifier.reclassify_file(
                    input_laz=laz_file,
                    output_laz=output_file,
                    ground_truth_features=ground_truth_features,
                )

                tiles_processed += 1
                logger.info(f"  ✅ Completed: {output_file.name}")

            except Exception as e:
                logger.error(f"  ❌ Failed: {e}", exc_info=True)
                tiles_skipped += 1
                continue

            # Garbage collection every 5 tiles
            if idx % 5 == 0:
                gc.collect()

        # Summary
        processing_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info("📊 Reclassification Summary:")
        logger.info(f"  Total files: {len(laz_files)}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  ⏱️  Time: {processing_time:.1f}s")
        logger.info("=" * 80)

        return tiles_processed
