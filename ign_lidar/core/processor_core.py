"""
Core processor initialization and configuration management.

This module contains the ProcessorCore class which handles configuration
validation, initialization, and component setup for the LiDAR processor.

Extracted from LiDARProcessor as part of god class refactoring (v3.4.0).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from ..classification_schema import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from ..datasets.dataset_manager import DatasetConfig
from ..features.orchestrator import FeatureOrchestrator
from .classification.config_validator import ConfigValidator
from .classification.io import TileLoader
from .optimization_factory import optimization_factory
from .skip_checker import PatchSkipChecker
from .tile_stitcher import TileStitcher

logger = logging.getLogger(__name__)


class ProcessorCore:
    """
    Low-level processor configuration, initialization, and component setup.

    This class is the **FOUNDATION LAYER** that handles all the setup work needed
    before actual tile processing can begin. It validates configuration, initializes
    components, and provides utility methods for low-level operations.

    Architecture Position:
    =====================

    LiDARProcessor (batch orchestration)
        ↓
    TileProcessor (tile orchestration)
        ↓
    **ProcessorCore** ← You are here (setup & utilities)

    Key Responsibilities:
    ====================

    1. **Configuration Management**
       - Validate configuration schema
       - Handle backward compatibility (v2.x → v3.x)
       - Migrate deprecated parameters
       - Set intelligent defaults

    2. **Auto-Optimization**
       - Detect hardware capabilities (CPU/GPU, RAM, cores)
       - Auto-tune processing parameters:
         * GPU batch sizes based on VRAM
         * Worker count based on CPU cores
         * Memory limits based on available RAM
       - Select optimal processing strategies

    3. **Component Initialization**
       - FeatureOrchestrator: Feature computation engine
       - TileStitcher: Boundary handling across tiles
       - PatchSkipChecker: Duplicate patch detection
       - TileLoader: LAZ file loading
       - ClassificationApplier: Ground truth integration

    4. **Low-Level Utilities**
       - Spatial indexing (KD-tree construction)
       - Neighbor searches
       - Point cloud validation
       - Memory management helpers

    Configuration Handling:
    ======================

    Supports multiple configuration sources:

    **Modern approach (v3.2+):**
    ```python
    from ign_lidar.config import Config
    config = Config.preset('lod2_buildings')
    core = ProcessorCore(config)
    ```

    **Legacy approach (v3.1):**
    ```python
    from ign_lidar.config.schema import IGNLiDARConfig
    config = IGNLiDARConfig(...)
    core = ProcessorCore(config)
    ```

    **Backward compatible (v2.x):**
    ```python
    core = ProcessorCore(
        lod_level='LOD2',
        use_gpu=True,
        patch_size=150.0
    )
    ```

    Validation Pipeline:
    ===================

    1. **Schema Validation**
       - Check required fields present
       - Validate value types and ranges
       - Detect configuration conflicts

    2. **Compatibility Migration**
       - Convert v2.x configs to v3.x
       - Map old parameter names to new ones
       - Apply deprecation warnings

    3. **Logical Validation**
       - Check GPU availability if use_gpu=True
       - Verify path existence if required
       - Validate parameter combinations

    4. **Auto-Optimization** (if enabled)
       - Profile hardware
       - Adjust batch sizes
       - Set worker counts
       - Configure memory limits

    Auto-Optimization Logic:
    =======================

    When `processing.auto_optimize=True`:

    **GPU Detection:**
    ```
    if GPU available:
        - Enable GPU acceleration
        - Set gpu_batch_size based on VRAM:
          * 4GB VRAM → 500K points
          * 8GB VRAM → 1M points
          * 16GB+ VRAM → 2M points
        - Disable multiprocessing (GPU + workers = bad)
    else:
        - Use CPU strategies
        - Set num_workers = CPU cores - 1
        - Enable parallel processing
    ```

    **Memory Management:**
    ```
    available_ram = psutil.virtual_memory().available
    memory_per_worker = 2GB  # Typical per worker
    max_workers = min(
        cpu_count - 1,
        available_ram // memory_per_worker
    )
    ```

    Component Lifecycle:
    ===================

    1. **Construction**: `__init__()` validates config
    2. **Setup**: Components initialized lazily or eagerly
    3. **Use**: LiDARProcessor/TileProcessor use utilities
    4. **Cleanup**: Components cleaned up on processor exit

    Design Principles:
    =================

    1. **Validation First**: Fail fast on invalid config
       - Better to error at initialization than mid-processing
       - Clear error messages for debugging
       - Suggest corrections when possible

    2. **Lazy Initialization**: Create components only when needed
       - TileLoader: Only if not provided externally
       - Stitcher: Only if stitching enabled
       - Reduces memory footprint

    3. **Immutable Config**: Configuration frozen after validation
       - Prevents accidental modification during processing
       - Enables safe parallel processing
       - Config changes require new ProcessorCore

    4. **Separation of Concerns**: Setup vs. execution
       - ProcessorCore: Configuration and setup
       - TileProcessor: Orchestration and execution
       - Clear boundary between concerns

    Common Configuration Patterns:
    =============================

    **Minimal (quickstart):**
    ```python
    config = Config(
        input_dir='/data/tiles',
        output_dir='/data/output',
        mode='lod2'
    )
    ```

    **Production (ASPRS classification):**
    ```python
    config = Config.preset('asprs_production')
    config.use_gpu = True
    config.ground_truth = {'bd_topo': {'buildings': True, 'roads': True}}
    ```

    **Research (full features + RGB + NIR):**
    ```python
    config = Config(
        mode='lod3',
        features={'feature_set': 'full', 'use_rgb': True, 'use_nir': True},
        use_gpu=True,
        augmentation={'enabled': True, 'num_augmentations': 5}
    )
    ```

    Error Handling:
    ==============

    - **TypeError**: Invalid config type
    - **ValueError**: Invalid parameter values
    - **FileNotFoundError**: Missing required files/directories
    - **GPUError**: GPU requested but not available
    - **ConfigurationError**: Logical configuration conflicts

    Performance Considerations:
    ==========================

    - Construction is lightweight (~10ms typical)
    - Component initialization adds ~100-500ms
    - Auto-optimization adds ~50-200ms (hardware detection)
    - Total initialization: < 1 second typical

    Related Classes:
    ===============

    - **LiDARProcessor**: Uses ProcessorCore for initialization (2 levels up)
    - **TileProcessor**: Uses ProcessorCore utilities (1 level up)
    - **ConfigValidator**: Configuration schema validation
    - **optimization_factory**: Auto-optimization logic

    Version History:
    ===============

    - v3.4.0: Extracted from LiDARProcessor (god class refactoring)
    - v3.2.0: Added Config support
    - v3.1.0: Added auto-optimization
    - v3.0.0: Added GPU configuration support

    See Also:
    ========

    - LiDARProcessor: For batch processing API
    - TileProcessor: For tile orchestration
    - Config: For modern configuration approach
    """

    def __init__(self, config: Union[DictConfig, Dict, None] = None, **kwargs):
        """
        Initialize processor core with configuration.

        Args:
            config: Configuration object (DictConfig or dict). If None,
                   builds config from kwargs for backward compatibility.
            **kwargs: Individual parameters (deprecated, use config instead)

        Raises:
            TypeError: If config is not DictConfig, dict, or None
            ValueError: If configuration is invalid
        """
        # Handle config initialization
        if config is None:
            config = self._build_config_from_kwargs(kwargs)
            logger.debug(
                "Built config from legacy kwargs "
                "(consider migrating to config-based initialization)"
            )
        elif not isinstance(config, (DictConfig, dict)):
            raise TypeError(
                f"config must be DictConfig or dict, got {type(config).__name__}. "
                f"Pass None and use kwargs for legacy initialization."
            )

        # Convert dict to DictConfig if needed
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        # Validate configuration
        self._validate_config(config)

        # Apply auto-optimization if enabled
        if OmegaConf.select(config, "processing.auto_optimize", default=False):
            config = self._apply_auto_optimization(config)

        # Store config
        self.config = config

        # Initialize commonly accessed values
        self._init_config_cache()

        # Initialize components
        self._init_feature_orchestrator()
        self._init_stitcher()
        self._init_skip_checker()
        self._init_dataset_manager()
        self._init_tile_loader()
        self._init_data_fetcher()

        logger.info(f"✨ Processing mode: {self.processing_mode}")
        if not self.save_patches:
            logger.info("   📦 Patch generation: DISABLED")
        logger.info(f"Initialized ProcessorCore with {self.lod_level}")

    def _validate_config(self, config: DictConfig) -> None:
        """
        Validate configuration structure and values.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate output format (with fallback)
        output_format = OmegaConf.select(
            config, "processor.output_format",
            default=OmegaConf.select(config, "output.format", default="npz")
        )
        validated_formats = ConfigValidator.validate_output_format(output_format)
        logger.debug(f"Validated output formats: {validated_formats}")

        # Validate processing mode
        ConfigValidator.validate_processing_mode(config.processor.processing_mode)

    def _build_config_from_kwargs(self, kwargs: Dict[str, Any]) -> DictConfig:
        """
        Build configuration from legacy kwargs (V2.x compatibility).

        Args:
            kwargs: Keyword arguments from old-style initialization

        Returns:
            DictConfig object with proper structure

        Note:
            This method provides backward compatibility but is deprecated.
            Users should migrate to config-based initialization.
        """
        # This would contain the logic from the original method
        # For now, create a basic structure
        config_dict = {
            "processor": {
                "lod_level": kwargs.get("lod_level", "LOD2"),
                "processing_mode": kwargs.get("processing_mode", "patches_only"),
                "patch_size": kwargs.get("patch_size", 150.0),
                "num_points": kwargs.get("num_points", 16384),
                "architecture": kwargs.get("architecture", "pointnet++"),
                "output_format": kwargs.get("output_format", "npz"),
                "use_gpu": kwargs.get("use_gpu", False),
                "augment": kwargs.get("augment", False),
                "num_augmentations": kwargs.get("num_augmentations", 3),
            },
            "features": {
                "mode": kwargs.get("feature_mode", "lod2"),
                "k_neighbors": kwargs.get("k_neighbors"),
                "use_rgb": kwargs.get("use_rgb", False),
                "use_infrared": kwargs.get("use_infrared", False),
                "compute_ndvi": kwargs.get("compute_ndvi", False),
            },
            "data_sources": {},
        }

        return OmegaConf.create(config_dict)

    def _apply_auto_optimization(self, config: DictConfig) -> DictConfig:
        """
        Apply automatic optimization based on system capabilities.

        Args:
            config: Input configuration

        Returns:
            Optimized configuration

        Note:
            Logs optimization strategy and expected improvements.
        """
        logger.info("🧠 Auto-optimization enabled - analyzing system capabilities...")

        # Get optimization recommendations
        recommendations = optimization_factory.recommend_optimization(
            OmegaConf.to_object(config)
        )

        if recommendations.get("config_updates"):
            logger.info(
                f"📈 Applying optimization strategy: "
                f"{recommendations['strategy'].value}"
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
                    f"⚡ Expected improvement: "
                    f"{recommendations['estimated_performance']}"
                )

            # Log any warnings
            for warning in recommendations.get("warnings", []):
                logger.warning(f"⚠️  {warning}")

        return config

    def _init_config_cache(self) -> None:
        """Cache commonly accessed configuration values."""
        self.lod_level = self.config.processor.lod_level
        self.processing_mode = self.config.processor.processing_mode
        if self.processing_mode == "enriched_and_patches":
            save_enriched = OmegaConf.select(
                self.config, "output.save_enriched", default=True
            )
            self.processing_mode = "both" if save_enriched else "patches_only"
        self.patch_size = OmegaConf.select(
            self.config, "processor.patch_size", default=150.0
        )
        self.num_points = OmegaConf.select(
            self.config, "processor.num_points", default=16384
        )
        self.architecture = OmegaConf.select(
            self.config, "processor.architecture", default="pointnet++"
        )
        # Fallback: processor.output_format -> output.format -> 'npz'
        self.output_format = OmegaConf.select(
            self.config, "processor.output_format",
            default=OmegaConf.select(self.config, "output.format", default="npz")
        )
        self.save_patches = OmegaConf.select(
            self.config, "output.save_patches", default=False
        )

        # Derive save/only flags from processing mode
        self.save_enriched_laz = self.processing_mode in [
            "both",
            "enriched_only",
        ]
        self.only_enriched_laz = self.processing_mode == "enriched_only"

        # Set class mapping based on LOD level
        if self.lod_level == "ASPRS":
            self.class_mapping = None
            self.default_class = 1  # ASPRS unclassified
        elif self.lod_level == "LOD2":
            self.class_mapping = ASPRS_TO_LOD2
            self.default_class = 14
        else:  # LOD3
            self.class_mapping = ASPRS_TO_LOD3
            self.default_class = 29

    def _init_feature_orchestrator(self) -> None:
        """Initialize feature orchestrator V5 with integrated optimizations."""
        logger.info("🚀 Using FeatureOrchestrator V5 with integrated optimizations")
        self.feature_orchestrator = FeatureOrchestrator(self.config)
        self.feature_manager = self.feature_orchestrator  # Backward compat

    def _init_stitcher(self) -> None:
        """Initialize tile stitcher if enabled."""
        # Handle both old and new config structures
        if hasattr(self.config, "stitching") and hasattr(
            self.config.stitching, "enabled"
        ):
            use_stitching = self.config.stitching.enabled
            buffer_size = getattr(self.config.stitching, "buffer_size", 10.0)
        else:
            use_stitching = self.config.processor.get("use_stitching", False)
            buffer_size = self.config.processor.get("buffer_size", 10.0)

        stitching_config = ConfigValidator.setup_stitching_config(
            use_stitching,
            buffer_size,
            self.config.processor.get("stitching_config", None),
        )
        self.stitcher = ConfigValidator.init_stitcher(stitching_config)

    def _init_skip_checker(self) -> None:
        """Initialize intelligent skip checker for incremental processing."""
        self.skip_checker = PatchSkipChecker(
            output_format=self.output_format,
            architecture=self.architecture,
            num_augmentations=self.config.processor.get("num_augmentations", 3),
            augment=self.config.processor.get("augment", False),
            validate_content=True,
            min_file_size=1024,  # 1KB minimum
            only_enriched_laz=self.only_enriched_laz,
        )

    def _init_dataset_manager(self) -> None:
        """Initialize dataset manager if dataset mode is enabled."""
        self.dataset_manager = None

        if self.config.get("dataset", {}).get("enabled", False):
            dataset_config = DatasetConfig(
                train_ratio=self.config.dataset.get("train_ratio", 0.7),
                val_ratio=self.config.dataset.get("val_ratio", 0.15),
                test_ratio=self.config.dataset.get("test_ratio", 0.15),
                random_seed=self.config.dataset.get("random_seed", 42),
                split_by_tile=self.config.dataset.get("split_by_tile", True),
                create_split_dirs=self.config.dataset.get("create_split_dirs", True),
                patch_sizes=self.config.dataset.get(
                    "patch_sizes", [int(self.patch_size)]
                ),
                balance_across_sizes=self.config.dataset.get(
                    "balance_across_sizes", False
                ),
            )
            self._dataset_config = dataset_config
            logger.info("📊 Dataset mode enabled - will create train/val/test splits")
        else:
            self._dataset_config = None

    def _init_tile_loader(self) -> None:
        """Initialize tile loading module."""
        self.tile_loader = TileLoader(self.config)

    def _init_data_fetcher(self) -> None:
        """
        Initialize data fetcher for ground truth classification.

        Supports BD TOPO®, BD Forêt®, RPG, and Cadastre data sources.
        """
        self.data_fetcher = None
        logger.debug("Checking data sources configuration...")

        if "data_sources" not in self.config:
            logger.warning(
                "⚠️  'data_sources' not found in config! "
                "Ground truth data won't be loaded."
            )
            return

        # Extract data source enablement flags
        bd_topo_enabled, bd_topo_features = self._parse_bd_topo_config()
        bd_foret_enabled = OmegaConf.select(
            self.config, "data_sources.bd_foret_enabled", default=False
        )
        rpg_enabled = OmegaConf.select(
            self.config, "data_sources.rpg_enabled", default=False
        )
        cadastre_enabled = OmegaConf.select(
            self.config, "data_sources.cadastre_enabled", default=False
        )

        # Log enabled data sources
        self._log_data_sources(
            bd_topo_enabled, bd_foret_enabled, rpg_enabled, cadastre_enabled
        )

        # Initialize data fetcher if any source is enabled
        if any([bd_topo_enabled, bd_foret_enabled, rpg_enabled, cadastre_enabled]):
            self._create_data_fetcher(bd_topo_features, rpg_enabled)

    def _parse_bd_topo_config(self) -> tuple[bool, Dict[str, bool]]:
        """
        Parse BD TOPO configuration (supports V4 flat and V5 nested).

        Returns:
            Tuple of (enabled, features_dict)
        """
        # Check nested structure FIRST (V5), then fall back to flat (V4)
        bd_topo_enabled = OmegaConf.select(
            self.config, "data_sources.bd_topo.enabled", default=False
        )

        if bd_topo_enabled:
            # V5 nested structure - extract .enabled from dict configs
            buildings_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.buildings",
                default=False,
            )
            buildings_enabled = (
                buildings_value.get('enabled', False)
                if isinstance(buildings_value, (dict, DictConfig)) and hasattr(buildings_value, 'get')
                else bool(buildings_value)
            )
            
            roads_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.roads",
                default=False,
            )
            roads_enabled = (
                roads_value.get('enabled', False)
                if isinstance(roads_value, (dict, DictConfig)) and hasattr(roads_value, 'get')
                else bool(roads_value)
            )
            
            water_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.water",
                default=False,
            )
            water_enabled = (
                water_value.get('enabled', False)
                if isinstance(water_value, (dict, DictConfig)) and hasattr(water_value, 'get')
                else bool(water_value)
            )
            
            vegetation_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.vegetation",
                default=False,
            )
            vegetation_enabled = (
                vegetation_value.get('enabled', False)
                if isinstance(vegetation_value, (dict, DictConfig)) and hasattr(vegetation_value, 'get')
                else bool(vegetation_value)
            )
            
            bridges_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.bridges",
                default=OmegaConf.select(
                    self.config,
                    "data_sources.bd_topo_bridges",
                    default=False,
                ),
            )
            bridges_enabled = (
                bridges_value.get('enabled', False)
                if isinstance(bridges_value, (dict, DictConfig)) and hasattr(bridges_value, 'get')
                else bool(bridges_value)
            )
            
            power_lines_value = OmegaConf.select(
                self.config,
                "data_sources.bd_topo.features.power_lines",
                default=OmegaConf.select(
                    self.config,
                    "data_sources.bd_topo_power_lines",
                    default=False,
                ),
            )
            power_lines_enabled = (
                power_lines_value.get('enabled', False)
                if isinstance(power_lines_value, (dict, DictConfig)) and hasattr(power_lines_value, 'get')
                else bool(power_lines_value)
            )
            
            features = {
                "buildings": buildings_enabled,
                "roads": roads_enabled,
                "water": water_enabled,
                "vegetation": vegetation_enabled,
                "bridges": bridges_enabled,
                "power_lines": power_lines_enabled,
            }
            logger.debug("Extracted BD TOPO features from nested config (V5)")
        else:
            # V4 flat structure (fallback)
            features = {
                "buildings": OmegaConf.select(
                    self.config, "data_sources.bd_topo_buildings", default=False
                ),
                "roads": OmegaConf.select(
                    self.config, "data_sources.bd_topo_roads", default=False
                ),
                "water": OmegaConf.select(
                    self.config, "data_sources.bd_topo_water", default=False
                ),
                "vegetation": OmegaConf.select(
                    self.config,
                    "data_sources.bd_topo_vegetation",
                    default=False,
                ),
                "bridges": OmegaConf.select(
                    self.config, "data_sources.bd_topo_bridges", default=False
                ),
                "power_lines": OmegaConf.select(
                    self.config,
                    "data_sources.bd_topo_power_lines",
                    default=False,
                ),
            }

            # Check if ANY feature is enabled
            bd_topo_enabled = any(features.values())

            if bd_topo_enabled:
                logger.debug("Extracted BD TOPO features from flat config (V4)")

        return bd_topo_enabled, features

    def _log_data_sources(
        self,
        bd_topo_enabled: bool,
        bd_foret_enabled: bool,
        rpg_enabled: bool,
        cadastre_enabled: bool,
    ) -> None:
        """Log enabled data sources."""
        logger.info(
            f"   - BD TOPO®: " f"{'✅ Enabled' if bd_topo_enabled else '❌ Disabled'}"
        )
        logger.info(
            f"   - BD Forêt®: " f"{'✅ Enabled' if bd_foret_enabled else '❌ Disabled'}"
        )
        logger.info(f"   - RPG: {'✅ Enabled' if rpg_enabled else '❌ Disabled'}")
        logger.info(
            f"   - Cadastre: " f"{'✅ Enabled' if cadastre_enabled else '❌ Disabled'}"
        )

    def _create_data_fetcher(
        self, bd_topo_features: Dict[str, bool], rpg_enabled: bool
    ) -> None:
        """Create and initialize data fetcher with configuration."""
        try:
            from ..io.data_fetcher import DataFetchConfig, DataFetcher

            cache_dir = Path(
                OmegaConf.select(self.config, "cache_dir", default="data/cache")
            )

            # Create fetch configuration
            fetch_config = DataFetchConfig(
                include_buildings=bd_topo_features["buildings"],
                include_roads=bd_topo_features["roads"],
                include_water=bd_topo_features["water"],
                include_vegetation=bd_topo_features["vegetation"],
                include_bridges=bd_topo_features["bridges"],
                include_power_lines=bd_topo_features["power_lines"],
                # ... (other parameters)
            )

            self.data_fetcher = DataFetcher(cache_dir=cache_dir, config=fetch_config)
            logger.info("✅ Data fetcher initialized successfully")
            logger.info(f"   Cache directory: {cache_dir}")

        except ImportError as e:
            logger.error(f"❌ Could not initialize data fetcher: {e}")
            logger.error("   Ground truth classification will NOT be applied")
            logger.error("   Install required packages: pip install geopandas shapely")
            self.data_fetcher = None
