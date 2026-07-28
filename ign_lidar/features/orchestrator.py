"""
Feature Orchestrator - Feature Computation System

Consolidates FeatureManager, FeatureComputer, and FeatureComputerFactory into a
single, cohesive orchestrator that manages all aspects of feature computation.

This module is part of Phase 4 consolidation effort to simplify the feature system
architecture and reduce code duplication.

Key Responsibilities:
- Resource Management: RGB, NIR fetchers and GPU validation
- Strategy Selection: CPU, GPU, Chunked, or Boundary-aware computation
- Feature Mode Enforcement: MINIMAL, LOD2, LOD3, FULL modes
- Computation Coordination: Geometric, spectral, and architectural features

Example:
    >>> from ign_lidar.features.orchestrator import FeatureOrchestrator
    >>> orchestrator = FeatureOrchestrator(config)
    >>> features = orchestrator.compute_features(tile_data)
    >>> print(f"Computed {len(features)} feature arrays")

Author: Phase 4 Consolidation
Date: October 13, 2025
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

# Error handling
from ..core.error_handler import (
    InitializationError,
    FeatureComputationError,
    CacheError,
    DataFetchError,
    GPUNotAvailableError,
    ConfigurationError,
)

# Multi-scale computation (v6.2)
from .compute.multi_scale import MultiScaleFeatureComputer, ScaleConfig

# Normalization utilities
from ..utils.normalization import normalize_rgb, normalize_nir, is_gpu_available

# NEW: Strategy Pattern (Week 2 refactoring)
from .strategies import BaseFeatureStrategy, FeatureComputeMode
from .strategy_cpu import CPUStrategy

try:
    from .strategy_gpu import GPUStrategy
    from .strategy_gpu_chunked import GPUChunkedStrategy
except ImportError:
    GPUStrategy = None
    GPUChunkedStrategy = None
from .feature_modes import FeatureMode, get_feature_config
from .strategy_boundary import BoundaryAwareStrategy

# Strategy pattern is now the standard - factory pattern removed


logger = logging.getLogger(__name__)

__all__ = ["FeatureOrchestrator"]


class FeatureOrchestrator:
    """
    Orchestrator for all feature computation operations.

    This class consolidates three previously separate concerns:
    1. Resource Management (RGB, NIR, GPU) - from FeatureManager
    2. Strategy Selection (CPU/GPU/Chunked) - from FeatureComputerFactory
    3. Computation Coordination - from FeatureComputer

    Benefits of consolidation:
    - Single entry point for all feature operations
    - Clear ownership of resources and strategy
    - Consistent feature mode enforcement
    - Easier to test and maintain
    - Less indirection and complexity

    Architecture:
        FeatureOrchestrator
            ├── Resource Management
            │   ├── rgb_fetcher: IGNOrthophotoFetcher
            │   ├── infrared_fetcher: IGNInfraredFetcher
            │   └── gpu_available: bool
            ├── Strategy Selection
            │   └── computer: BaseFeatureComputer
            ├── Mode Management
            │   └── feature_mode: FeatureMode
            └── Computation Methods
                ├── compute_features()
                ├── compute_geometric_features()
                └── add_spectral_features()

    Example:
        >>> config = OmegaConf.load("config.yaml")
        >>> orchestrator = FeatureOrchestrator(config)
        >>>
        >>> # Compute features for a tile
        >>> features = orchestrator.compute_features(tile_data)
        >>>
        >>> # Check what was computed
        >>> print(f"Features: {list(features.keys())}")
        >>> print(f"Strategy: {orchestrator.strategy_name}")
        >>> print(f"Mode: {orchestrator.feature_mode}")
    """

    def __init__(
        self, config: DictConfig, progress_callback: Optional[callable] = None
    ):
        """
        Initialize feature orchestrator from configuration.

        This performs three main initialization steps:
        1. Initialize external resources (RGB, NIR, GPU)
        2. Select and create appropriate feature computer
        3. Setup feature mode and validation

        Args:
            config: Hydra/OmegaConf configuration object with sections:
                - processor: use_gpu, use_gpu_chunked, etc.
                - features: mode, k_neighbors, use_rgb, use_infrared, etc.
            progress_callback: Optional callback function(progress: float,
                message: str) for progress updates during computation

        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing
        """
        self.config = config
        self.progress_callback = progress_callback

        # Initialize resources (RGB, NIR, GPU)
        self._init_resources()

        # Select and create feature computer
        self._init_computer()

        # Setup feature mode
        self._init_feature_mode()

        # Initialize multi-scale computation if enabled
        self._init_multi_scale()

        # ✅ OPTIMIZATION: Cache frequently accessed config values
        self._cache_config_values()

        # ✅ OPTIMIZATION: Initialize advanced optimizations
        self._init_optimizations()

        # Log feature configuration with data availability
        self._log_feature_config()

        logger.info(
            f"FeatureOrchestrator V5 initialized | "
            f"strategy={self.strategy_name} | "
            f"mode={self.feature_mode} | "
            f"gpu={self.gpu_available}"
        )

    # =========================================================================
    # RESOURCE MANAGEMENT (from FeatureManager)
    # =========================================================================

    def _init_resources(self):
        """
        Initialize external resources for feature computation.

        This includes:
        - RGB orthophoto fetcher (if use_rgb enabled)
        - Infrared (NIR) fetcher (if use_infrared enabled)
        - GPU availability validation (if use_gpu enabled)

        Resources are initialized lazily - they're only created if actually
        needed based on configuration.
        """
        processor_cfg = self.config.get("processor", {})
        features_cfg = self.config.get("features", {})

        # Initialize RGB fetcher if needed
        self.use_rgb = features_cfg.get("use_rgb", False)
        self.rgb_fetcher = None
        if self.use_rgb:
            self.rgb_fetcher = self._init_rgb_fetcher()

        # Initialize NIR fetcher if needed
        # Support both 'use_infrared' (current) and 'use_nir' (legacy) for backward compatibility
        self.use_infrared = features_cfg.get(
            "use_infrared", features_cfg.get("use_nir", False)
        )
        self.infrared_fetcher = None
        if self.use_infrared:
            self.infrared_fetcher = self._init_infrared_fetcher()

        # Validate GPU availability if needed
        self.use_gpu = processor_cfg.get("use_gpu", False)
        self.gpu_available = False
        if self.use_gpu:
            self.gpu_available = self._validate_gpu()

        logger.debug(
            f"Resources initialized | "
            f"rgb={self.rgb_fetcher is not None} | "
            f"nir={self.infrared_fetcher is not None} | "
            f"gpu={self.gpu_available}"
        )

    def _init_rgb_fetcher(self):
        """
        Initialize RGB orthophoto fetcher.

        Returns:
            IGNOrthophotoFetcher instance or None if initialization fails
        """
        try:
            from ..preprocessing.rgb_augmentation import IGNOrthophotoFetcher

            # Determine cache directory
            rgb_cache_dir = self.config.features.get("rgb_cache_dir")
            if rgb_cache_dir is None:
                rgb_cache_dir = (
                    Path(tempfile.gettempdir()) / "ign_lidar_cache" / "orthophotos"
                )
                rgb_cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                rgb_cache_dir = Path(rgb_cache_dir)

            fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
            logger.info(
                "RGB enabled (will use from input LAZ if present, "
                "otherwise fetch from IGN orthophotos)"
            )
            return fetcher

        except ImportError as e:
            error = InitializationError.create(
                component="RGB fetcher",
                error=e,
                dependencies=["requests", "Pillow"]
            )
            logger.error(str(error))
            return None
        except (OSError, IOError) as e:
            logger.error(f"Failed to create RGB cache directory: {e}")
            logger.error(f"  Check write permissions for: {rgb_cache_dir}")
            return None
        except Exception as e:
            error = InitializationError.create(
                component="RGB fetcher",
                error=e
            )
            logger.error(str(error))
            return None

    def _init_infrared_fetcher(self):
        """
        Initialize infrared (NIR) fetcher.

        Returns:
            IGNInfraredFetcher instance or None if initialization fails
        """
        try:
            from ..preprocessing.infrared_augmentation import IGNInfraredFetcher

            # Determine cache directory
            rgb_cache_dir = self.config.features.get("rgb_cache_dir")
            if rgb_cache_dir is None:
                infrared_cache_dir = (
                    Path(tempfile.gettempdir()) / "ign_lidar_cache" / "infrared"
                )
            else:
                infrared_cache_dir = Path(rgb_cache_dir).parent / "infrared"

            infrared_cache_dir.mkdir(parents=True, exist_ok=True)

            fetcher = IGNInfraredFetcher(cache_dir=infrared_cache_dir)
            logger.info(
                "NIR enabled (will use from input LAZ if present, "
                "otherwise fetch from IGN IRC)"
            )
            return fetcher

        except ImportError as e:
            error = InitializationError.create(
                component="NIR fetcher",
                error=e,
                dependencies=["requests", "Pillow"]
            )
            logger.error(str(error))
            return None
        except (OSError, IOError) as e:
            logger.error(f"Failed to create NIR cache directory: {e}")
            logger.error(f"  Check write permissions for: {infrared_cache_dir}")
            return None
        except Exception as e:
            error = InitializationError.create(
                component="NIR fetcher",
                error=e
            )
            logger.error(str(error))
            return None

    def _validate_gpu(self) -> bool:
        """
        Validate GPU availability.

        Returns:
            bool: True if GPU is available and working
        """
        try:
            from .gpu_processor import GPU_AVAILABLE

            if not GPU_AVAILABLE:
                logger.warning("GPU requested but CuPy not available. Using CPU.")
                return False

            logger.info("GPU acceleration enabled")
            return True

        except ImportError as e:
            logger.warning("GPU module not available. Using CPU.")
            logger.debug(f"  Import error details: {e}")
            return False
        except Exception as e:
            error = GPUNotAvailableError.create(reason=str(e))
            logger.warning(str(error))
            return False

    def _init_multi_scale(self):
        """
        Initialize multi-scale feature computation (v6.2).

        Detects if multi-scale is enabled in config and creates
        MultiScaleFeatureComputer instance if needed.

        Multi-scale computation:
        - Computes features at multiple neighborhood scales
        - Aggregates using variance weighting to suppress artifacts
        - Reduces scan line artifacts by 20-40% → 5-10%
        """
        features_cfg = self.config.get("features", {})

        # Check if multi-scale is enabled
        self.use_multi_scale = features_cfg.get("multi_scale_computation", False)
        self.multi_scale_computer = None

        if not self.use_multi_scale:
            return

        # Validate that scales are configured
        scales_config = features_cfg.get("scales", [])
        if not scales_config or len(scales_config) < 2:
            logger.warning(
                "Multi-scale computation enabled but scales not "
                "configured. Disabling multi-scale."
            )
            self.use_multi_scale = False
            return

        # Parse scale configurations
        try:
            scales = []
            for scale_cfg in scales_config:
                scale = ScaleConfig(
                    name=scale_cfg["name"],
                    k_neighbors=scale_cfg["k_neighbors"],
                    search_radius=scale_cfg["search_radius"],
                    weight=scale_cfg.get("weight", 1.0),
                )
                scales.append(scale)

            # Create multi-scale computer
            aggregation_method = features_cfg.get(
                "aggregation_method", "variance_weighted"
            )
            variance_penalty = features_cfg.get("variance_penalty_factor", 2.0)

            self.multi_scale_computer = MultiScaleFeatureComputer(
                scales=scales,
                aggregation_method=aggregation_method,
                variance_penalty=variance_penalty,
            )

            logger.info(
                f"  🔬 Multi-scale computation enabled | "
                f"scales={len(scales)} | "
                f"method={aggregation_method}"
            )
            
            # Connect GPU processor if available (must be done after _init_computer)
            self._connect_multi_scale_gpu()

        except (ValueError, KeyError) as e:
            logger.error(f"Invalid multi-scale configuration: {e}")
            logger.error("  Check scales configuration in features section")
            self.use_multi_scale = False
            self.multi_scale_computer = None
        except ImportError as e:
            error = InitializationError.create(
                component="multi-scale computer",
                error=e
            )
            logger.error(str(error))
            self.use_multi_scale = False
            self.multi_scale_computer = None
        except Exception as e:
            error = InitializationError.create(
                component="multi-scale computer",
                error=e
            )
            logger.error(str(error))
            self.use_multi_scale = False
            self.multi_scale_computer = None
    
    def _connect_multi_scale_gpu(self):
        """
        Connect GPU processor to multi-scale computer after initialization.
        
        This performs intelligent GPU connection with validation and fallback:
        1. Validates GPU availability when requested
        2. Tests GPU functionality with CuPy
        3. Provides actionable error messages
        4. Falls back to CPU gracefully
        
        This must be called after _init_computer() to ensure self.computer exists.
        """
        if not self.use_multi_scale or self.multi_scale_computer is None:
            return
        
        features_cfg = self.config.get("features", {})
        use_gpu = features_cfg.get("use_gpu", False) or features_cfg.get("force_gpu", False)
        
        if not use_gpu:
            logger.debug("  💻 Multi-scale using CPU (GPU not requested)")
            return
        
        # Try to get GPU processor from strategy
        gpu_processor = None
        if hasattr(self.computer, 'gpu_processor'):
            gpu_processor = self.computer.gpu_processor
        
        if gpu_processor is None:
            logger.warning("  ⚠️  GPU requested but no GPU processor available in feature computer")
            logger.warning("       This may occur if:")
            logger.warning("       1. CuPy is not installed (install: pip install cupy-cuda11x or cupy-cuda12x)")
            logger.warning("       2. No CUDA-compatible GPU detected")
            logger.warning("       3. GPU initialization failed during feature computer setup")
            logger.warning("       → Multi-scale features will use CPU fallback")
            logger.info("  💻 Multi-scale falling back to CPU mode")
            self.multi_scale_computer.use_gpu = False
            return
        
        # Validate GPU functionality
        try:
            import cupy as cp
            
            # Test GPU with small operation
            test_array = cp.array([1.0, 2.0, 3.0])
            _ = cp.mean(test_array)
            cp.cuda.Stream.null.synchronize()
            
            # GPU is working - connect it
            self.multi_scale_computer.use_gpu = True
            self.multi_scale_computer.gpu_processor = gpu_processor
            
            # Get GPU info for logging
            device = cp.cuda.Device()
            gpu_name = device.name
            mem_info = device.mem_info
            mem_total_gb = mem_info[1] / 1e9
            mem_free_gb = mem_info[0] / 1e9
            
            logger.info(f"  🚀 Multi-scale connected to GPU: {gpu_name}")
            logger.info(f"       GPU memory: {mem_free_gb:.1f}/{mem_total_gb:.1f} GB free")
            
        except ImportError:
            logger.error("  ❌ GPU requested but CuPy not installed!")
            logger.error("       Install with: conda install -c conda-forge cupy")
            logger.error("       or: pip install cupy-cuda11x (for CUDA 11.x)")
            logger.error("       or: pip install cupy-cuda12x (for CUDA 12.x)")
            logger.info("  💻 Multi-scale falling back to CPU mode")
            self.multi_scale_computer.use_gpu = False
            
        except Exception as e:
            logger.error(f"  ❌ GPU validation failed: {e}")
            logger.error("       This may indicate:")
            logger.error("       1. GPU out of memory")
            logger.error("       2. CUDA driver issues")
            logger.error("       3. GPU hardware problems")
            logger.info("  💻 Multi-scale falling back to CPU mode")
            self.multi_scale_computer.use_gpu = False

    # =========================================================================
    # STRATEGY SELECTION (from FeatureComputerFactory)
    # =========================================================================

    def _init_computer(self):
        """
        Select and create appropriate feature strategy based on configuration.

        NEW (Phase 4 Task 1.4): Supports FeatureComputer with automatic mode selection

        Selection logic:
        1. If use_feature_computer=True: Use FeatureComputer (automatic mode selection)
        2. Else: Use Strategy Pattern (manual GPU/CPU selection) - via _init_strategy_computer()

        The created strategy is cached in self.computer for reuse.

        Note: self.computer can be:
        - FeatureComputer (Phase 4, automatic mode selection)
        - BaseFeatureStrategy (Week 2, Strategy Pattern)
        - BaseFeatureComputer (legacy, deprecated)
        """
        processor_cfg = self.config.get("processor", {})

        # NEW (Phase 4 Task 1.4): Check if feature computer is enabled
        use_feature_computer = processor_cfg.get("use_feature_computer", False)

        if use_feature_computer:
            logger.info("🆕 Using FeatureComputer (Phase 4 - automatic mode selection)")
            self._init_feature_computer()
        else:
            # Use legacy strategy pattern (original implementation)
            self._init_strategy_computer()

    def _init_feature_computer(self):
        """
        Initialize FeatureComputer with automatic mode selection.

        NEW (Phase 4 Task 1.4): Provides automatic mode selection based on:
        - Point cloud size
        - GPU availability
        - Memory constraints
        - User configuration overrides

        The FeatureComputer provides a single, consistent API across
        all computation modes (CPU, GPU, GPU_CHUNKED, BOUNDARY).
        """
        try:
            from .feature_computer import FeatureComputer
            from .mode_selector import ModeSelector
        except ImportError as e:
            error = InitializationError.create(
                component="FeatureComputer",
                error=e,
                dependencies=["feature_computer", "mode_selector"]
            )
            logger.error(str(error))
            # Fall back to strategy pattern
            self._init_strategy_computer()
            return

        # Get forced mode from config (if any)
        force_mode_str = self._get_forced_mode_from_config()

        # Convert mode string to ComputationMode enum if specified
        force_mode = None
        if force_mode_str:
            from .mode_selector import ComputationMode

            mode_map = {
                "cpu": ComputationMode.CPU,
                "gpu": ComputationMode.GPU,
                "gpu_chunked": ComputationMode.GPU_CHUNKED,
                "boundary": ComputationMode.BOUNDARY,
            }
            force_mode = mode_map.get(force_mode_str.lower())

        # Create feature computer
        prefer_gpu = self.gpu_available
        self.computer = FeatureComputer(
            mode_selector=None,  # Use default mode selector
            force_mode=force_mode,
            progress_callback=self.progress_callback,
            prefer_gpu=prefer_gpu,
        )

        # Set strategy name for logging
        if force_mode:
            self.strategy_name = force_mode
            logger.info(f"   📌 Mode: {force_mode} (forced by config)")
        else:
            self.strategy_name = "auto"
            logger.info("   📌 Mode: automatic (intelligent selection)")

        # Log mode recommendations for typical tile size
        try:
            selector = ModeSelector()
            typical_size = self._estimate_typical_tile_size()
            recommendations = selector.get_recommendations(num_points=typical_size)
            logger.info(
                f"   💡 Recommended for {typical_size:,} points: "
                f"{recommendations['recommended_mode']}"
            )
        except Exception as e:
            logger.debug(f"Could not get mode recommendations: {e}")

    def _get_forced_mode_from_config(self) -> Optional[str]:
        """
        Get forced computation mode from configuration.

        Maps legacy config flags to computer modes for backward compatibility.

        Returns:
            str or None: Forced mode ('cpu', 'gpu', 'gpu_chunked', 'boundary') or None for auto

        Config priority:
            1. processor.computation_mode (new, explicit)
            2. Legacy flags (use_gpu, use_gpu_chunked, use_boundary_aware)
            3. None (automatic mode selection)
        """
        processor_cfg = self.config.get("processor", {})

        # Check for explicit computation_mode setting (highest priority)
        if "computation_mode" in processor_cfg:
            mode = processor_cfg["computation_mode"]
            if mode.lower() in ["auto", "automatic"]:
                return None  # Automatic mode
            return mode.lower()

        # Map legacy flags to modes (for backward compatibility)
        if processor_cfg.get("use_boundary_aware", False):
            return "boundary"

        if processor_cfg.get("use_gpu", False):
            # Check if chunked mode is explicitly disabled
            use_chunked = processor_cfg.get(
                "use_gpu_chunked", True
            )  # Default to chunked
            if use_chunked:
                return "gpu_chunked"
            else:
                return "gpu"

        # No forced mode - use automatic selection
        return None

    def _estimate_typical_tile_size(self) -> int:
        """
        Estimate typical tile size for mode recommendations.

        Returns:
            int: Estimated number of points per tile

        Note:
            This is a rough estimate. Actual tile sizes vary significantly.
            Typical IGN LIDAR HD tiles range from 500K to 10M points.
        """
        # Use config hints if available
        processor_cfg = self.config.get("processor", {})

        # If patch processing, estimate from patch params
        if "patch_size" in processor_cfg:
            patch_size = processor_cfg["patch_size"]  # meters
            # Rough estimate: ~100-200 points per square meter for IGN LIDAR HD
            points_per_sqm = 150
            estimated_points = int(patch_size * patch_size * points_per_sqm)
            return estimated_points

        # Default estimate for full tiles
        # Most IGN tiles are 1-5M points
        return 2_000_000

    def _init_strategy_computer(self):
        """
        Initialize using legacy Strategy Pattern (for backward compatibility).

        This is the original strategy selection logic, preserved for:
        1. Backward compatibility
        2. Fallback if FeatureComputer unavailable
        3. Gradual migration path

        Note: This will be called by _init_computer() when use_feature_computer=False
        """
        processor_cfg = self.config.get("processor", {})
        features_cfg = self.config.get("features", {})

        # Extract parameters
        k_neighbors = features_cfg.get("k_neighbors", 20)
        radius = features_cfg.get("search_radius", 1.0)
        use_boundary_aware = processor_cfg.get("use_boundary_aware", False)

        # GPU chunked strategy selection:
        # Default to TRUE when GPU available (most optimized strategy)
        # Can be explicitly disabled by setting use_gpu_chunked=False in config
        # Check both features and processor configs (features takes precedence)
        use_gpu_chunked_config = features_cfg.get(
            "use_gpu_chunked", processor_cfg.get("use_gpu_chunked", None)
        )
        if use_gpu_chunked_config is None:
            # Not specified in config - use intelligent default
            # Default to chunked when GPU available (most optimized)
            use_gpu_chunked = self.gpu_available and (GPUChunkedStrategy is not None)
        else:
            # Respect explicit config setting
            use_gpu_chunked = use_gpu_chunked_config

        use_strategy_pattern = processor_cfg.get(
            "use_strategy_pattern", True
        )  # NEW: opt-in

        # Initialize size tracking for logging
        gpu_size = None

        # NEW (Week 2): Strategy Pattern implementation
        if use_strategy_pattern:
            logger.info("🆕 Using Strategy Pattern (Week 2 refactoring)")

            # Select base strategy
            if self.gpu_available and use_gpu_chunked:
                self.strategy_name = "gpu_chunked"
                chunk_size = processor_cfg.get("gpu_batch_size", 5_000_000)
                batch_size = 250_000  # Week 1 optimized batch size
                neighbor_query_batch_size = features_cfg.get(
                    "neighbor_query_batch_size", None
                )
                feature_batch_size = features_cfg.get("feature_batch_size", None)
                gpu_size = chunk_size

                # Log whether this is default or explicit choice
                if use_gpu_chunked_config is None:
                    logger.info(
                        "   📌 Using GPU chunked strategy (intelligent default - most optimized)"
                    )
                else:
                    logger.info(
                        "   📌 Using GPU chunked strategy (explicitly configured)"
                    )

                if GPUChunkedStrategy is None:
                    logger.warning(
                        "GPU chunked strategy not available, falling back to CPU"
                    )
                    base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)
                else:
                    base_strategy = GPUChunkedStrategy(
                        k_neighbors=k_neighbors,
                        radius=radius,
                        chunk_size=chunk_size,
                        batch_size=batch_size,
                        neighbor_query_batch_size=neighbor_query_batch_size,
                        feature_batch_size=feature_batch_size,
                    )

            elif self.gpu_available:
                self.strategy_name = "gpu"
                batch_size = features_cfg.get(
                    "gpu_batch_size", processor_cfg.get("gpu_batch_size", 8_000_000)
                )
                gpu_size = batch_size

                if GPUStrategy is None:
                    logger.warning("GPU strategy not available, falling back to CPU")
                    base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)
                else:
                    base_strategy = GPUStrategy(
                        k_neighbors=k_neighbors, radius=radius, batch_size=batch_size
                    )
            else:
                self.strategy_name = "cpu"
                base_strategy = CPUStrategy(k_neighbors=k_neighbors, radius=radius)

            # Wrap with boundary-aware strategy if needed
            if use_boundary_aware:
                buffer_size = processor_cfg.get("buffer_size", 10.0)
                self.computer = BoundaryAwareStrategy(
                    base_strategy=base_strategy, boundary_buffer=buffer_size
                )
                self.strategy_name = f"boundary_aware({self.strategy_name})"
            else:
                self.computer = base_strategy

        logger.debug(f"Selected strategy: {self.strategy_name}")
        if gpu_size is not None:
            logger.info(f"  💾 GPU batch/chunk size: {gpu_size:,} points")

    def _cache_config_values(self):
        """
        Cache frequently accessed config values to avoid repeated lookups.

        OPTIMIZATION: Phase 1 - Quick Win
        - Caches ~10 most frequently accessed values
        - Reduces dict lookup overhead (~50-100ns per lookup)
        - Saves ~10ms total across thousands of accesses
        - Negligible impact individually, but good practice

        This is a micro-optimization but demonstrates proper config handling.
        """
        features_cfg = self.config.get("features", {})
        processor_cfg = self.config.get("processor", {})

        # Cache common values
        self._k_neighbors_cached = features_cfg.get("k_neighbors", 20)
        self._use_gpu_chunked_cached = processor_cfg.get("use_gpu_chunked", False)
        self._gpu_batch_size_cached = features_cfg.get("gpu_batch_size", 1_000_000)
        self._search_radius_cached = features_cfg.get("search_radius", 1.0)
        self._use_boundary_aware_cached = processor_cfg.get("use_boundary_aware", False)
        self._buffer_size_cached = processor_cfg.get("buffer_size", 10.0)

        logger.debug("Config values cached for fast access")

    # =========================================================================
    # FEATURE MODE MANAGEMENT
    # =========================================================================

    def _init_feature_mode(self):
        """
        Initialize and validate feature mode.

        Feature modes control which features are computed:
        - MINIMAL: ~8 features (fast)
        - LOD2_SIMPLIFIED: ~11 features (building classification)
        - LOD3_FULL: ~35 features (detailed modeling)
        - FULL: All available features
        - CUSTOM: User-defined selection
        """
        features_cfg = self.config.get("features", {})
        # Check both 'mode' (Hydra configs) and 'feature_mode' (legacy kwargs)
        # Use 'lod2' as default if neither is specified
        mode_str = (
            features_cfg.get("mode") or features_cfg.get("feature_mode") or "lod2"
        )

        # Debug: Log the mode being loaded (with full config inspection)
        logger.info(f"🔍 DEBUG: features_cfg = {dict(features_cfg)}")
        logger.info(
            f"🔍 DEBUG: mode_str = '{mode_str}' (type: {type(mode_str).__name__})"
        )

        # Parse mode
        try:
            if isinstance(mode_str, FeatureMode):
                self.feature_mode = mode_str
                logger.info(
                    f"🔍 DEBUG: mode_str is already a FeatureMode: {self.feature_mode}"
                )
            else:
                self.feature_mode = FeatureMode(mode_str.lower())
                logger.info(
                    f"🔍 DEBUG: Parsed mode string '{mode_str}' to {self.feature_mode}"
                )
        except ValueError:
            logger.warning(f"Invalid feature mode '{mode_str}', using FULL")
            self.feature_mode = FeatureMode.FULL

        # Get feature configuration for this mode
        # Note: We don't know data availability yet during initialization,
        # so we suppress the detailed logging here (will be shown during processing)
        self.feature_config = get_feature_config(
            self.feature_mode.value,
            has_rgb=None,
            has_nir=None,
            log_config=False,  # Suppress logging during init
        )

        logger.debug(
            f"Feature mode: {self.feature_mode.value} | "
            f"features: {len(self.feature_config.feature_names)}"
        )

    def _log_feature_config(self):
        """Log feature configuration with data availability information."""
        # Check if RGB/NIR fetchers are available (not necessarily that data exists yet)
        has_rgb_fetcher = self.rgb_fetcher is not None
        has_nir_fetcher = self.infrared_fetcher is not None

        logger.info(
            f"📊 Feature Configuration: {self.feature_config.get_description()}"
        )
        logger.info(f"   Features: {', '.join(self.feature_config.feature_names)}")

        # Log RGB/NIR status based on whether fetchers are available
        if self.feature_config.requires_rgb:
            if has_rgb_fetcher or self.use_rgb:
                logger.info(
                    "   ✓ RGB channels enabled (will use from input LAZ or fetch if needed)"
                )
            elif has_rgb_fetcher is False and not self.use_rgb:
                logger.debug(
                    "   ⚠️  RGB channels required but RGB fetcher not available (will attempt to load from LAZ)"
                )

        if self.feature_config.requires_nir:
            if has_nir_fetcher or self.use_infrared:
                logger.info(
                    "   ✓ NIR channel enabled (will use from input LAZ or fetch if needed)"
                )
            elif has_nir_fetcher is False and not self.use_infrared:
                logger.debug(
                    "   ⚠️  NIR channel required but NIR fetcher not available (will attempt to load from LAZ)"
                )

    # =========================================================================
    # OPTIMIZATION FEATURES (V5 Consolidation)
    # =========================================================================

    def _init_optimizations(self):
        """Initialize optimization subsystems for feature computation."""
        self._init_caching()
        self._init_parallel_processing()
        self._init_adaptive_parameters()
        self._init_performance_monitoring()

        logger.debug("FeatureOrchestrator V5 optimizations initialized")

    def _init_caching(self):
        """Initialize intelligent feature caching system."""
        features_cfg = self.config.get("features", {})

        self._enable_feature_cache = features_cfg.get("enable_caching", True)
        self._cache_max_size = features_cfg.get("cache_max_size", 100)  # MB
        self._feature_cache = {}
        self._current_cache_size = 0
        
        # ✅ NEW (v3.5.2): Intermediate result cache for normals & eigenvalues
        # Avoids recomputation when multiple features need them
        self._intermediate_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if self._enable_feature_cache:
            logger.debug("Feature caching enabled (includes intermediate results)")

    def _init_parallel_processing(self):
        """Initialize parallel processing for RGB/NIR operations."""
        from concurrent.futures import ThreadPoolExecutor

        processor_cfg = self.config.get("processor", {})
        num_workers = processor_cfg.get("num_workers", 1)

        # Create thread pools for different operations
        self._rgb_nir_executor = ThreadPoolExecutor(
            max_workers=min(4, max(1, num_workers // 2)), thread_name_prefix="rgb_nir"
        )
        self._feature_executor = ThreadPoolExecutor(
            max_workers=min(2, max(1, num_workers)), thread_name_prefix="features"
        )

        logger.debug(
            f"Parallel processing initialized: "
            f"rgb_nir_workers={self._rgb_nir_executor._max_workers}, "
            f"feature_workers={self._feature_executor._max_workers}"
        )

    def _init_adaptive_parameters(self):
        """Initialize adaptive parameter tuning."""
        features_cfg = self.config.get("features", {})

        # Base parameters that can be adapted
        self._adaptive_parameters = {
            "k_neighbors": features_cfg.get("k_neighbors", 20),
            "search_radius": features_cfg.get("search_radius", 1.0),
            "batch_size": features_cfg.get("gpu_batch_size", 1_000_000),
        }

        # Adaptation history
        self._parameter_history = {}
        self._adaptation_enabled = features_cfg.get("enable_auto_tuning", True)

        if self._adaptation_enabled:
            logger.debug("Adaptive parameter tuning enabled")

    def _init_performance_monitoring(self):
        """Initialize performance monitoring and metrics collection."""
        monitoring_cfg = self.config.get("monitoring", {})

        self._enable_profiling = monitoring_cfg.get("enable_profiling", False)
        self._enable_performance_metrics = monitoring_cfg.get(
            "enable_performance_metrics", True
        )

        # Performance tracking
        self._processing_times = []
        self._memory_usage = []
        self._gpu_utilization = []

        if self._enable_performance_metrics:
            logger.debug("Performance monitoring enabled")

    def _generate_cache_key(self, points, classification, kwargs):
        """Generate a unique cache key for the given inputs."""
        import hashlib

        # Create a hash from point statistics and parameters
        data_hash = hashlib.md5()
        data_hash.update(str(points.shape).encode())
        data_hash.update(str(np.mean(points, axis=0)).encode())
        data_hash.update(str(np.std(points, axis=0)).encode())
        data_hash.update(str(np.unique(classification)).encode())
        data_hash.update(str(sorted(kwargs.items())).encode())

        return data_hash.hexdigest()
    
    def _get_cached_normals_eigenvalues(self, points: np.ndarray, k_neighbors: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get cached normals and eigenvalues if available.
        
        ✅ NEW (v3.5.2): Avoids recomputing normals/eigenvalues when multiple
        features (curvature, planarity, linearity, etc.) need them.
        
        Args:
            points: Point cloud array (N, 3)
            k_neighbors: Number of neighbors used in computation
            
        Returns:
            Tuple of (normals, eigenvalues) if cached, None otherwise
        """
        if not self._enable_feature_cache:
            return None
            
        # Generate cache key based on points hash and k_neighbors
        import hashlib
        cache_key = hashlib.md5(
            f"{points.shape}_{np.mean(points)}_{np.std(points)}_{k_neighbors}".encode()
        ).hexdigest()
        
        if cache_key in self._intermediate_cache:
            self._cache_hits += 1
            logger.debug(f"✅ Cache HIT for normals/eigenvalues (hits={self._cache_hits}, misses={self._cache_misses})")
            return self._intermediate_cache[cache_key]
        else:
            self._cache_misses += 1
            return None
    
    def _cache_normals_eigenvalues(self, points: np.ndarray, k_neighbors: int, 
                                   normals: np.ndarray, eigenvalues: np.ndarray) -> None:
        """
        Cache computed normals and eigenvalues for reuse.
        
        ✅ NEW (v3.5.2): Stores intermediate results to avoid recomputation.
        
        Args:
            points: Point cloud array (N, 3)
            k_neighbors: Number of neighbors used
            normals: Computed normal vectors (N, 3)
            eigenvalues: Computed eigenvalues (N, 3)
        """
        if not self._enable_feature_cache:
            return
            
        # Generate same cache key as in _get_cached_normals_eigenvalues
        import hashlib
        cache_key = hashlib.md5(
            f"{points.shape}_{np.mean(points)}_{np.std(points)}_{k_neighbors}".encode()
        ).hexdigest()
        
        # Store in cache (will be cleared when object is destroyed)
        self._intermediate_cache[cache_key] = (normals.copy(), eigenvalues.copy())
        
        # Limit cache size to prevent memory bloat
        if len(self._intermediate_cache) > 10:
            # Remove oldest entry (simple FIFO)
            self._intermediate_cache.pop(next(iter(self._intermediate_cache)))
            logger.debug("Intermediate cache size limited (removed oldest entry)")

    def _optimize_parameters_for_data(self, points, classification):
        """
        Optimize parameters based on data characteristics.

        Args:
            points: Point cloud data
            classification: Classification codes

        Returns:
            Optimized parameters
        """
        n_points = len(points)

        # Analyze point density
        bbox_volume = self._estimate_point_cloud_volume(points)
        point_density = n_points / bbox_volume if bbox_volume > 0 else 1000

        # Base parameters
        optimized = self._adaptive_parameters.copy()

        # Optimize k_neighbors based on density
        base_k = optimized["k_neighbors"]
        if point_density > 10000:  # Very dense
            optimized["k_neighbors"] = min(base_k + 10, 50)
        elif point_density > 1000:  # Dense
            optimized["k_neighbors"] = base_k
        else:  # Sparse
            optimized["k_neighbors"] = max(base_k - 5, 10)

        # Optimize search radius based on data distribution
        z_range = np.ptp(points[:, 2])  # Z-axis range
        base_radius = optimized["search_radius"]
        if z_range > 100:  # Building/urban area
            optimized["search_radius"] = base_radius * 1.5
        elif z_range < 10:  # Flat area
            optimized["search_radius"] = base_radius * 0.8

        # Optimize batch size for GPU processing
        base_batch_size = optimized["batch_size"]
        if n_points > 10_000_000:  # Very large
            optimized["batch_size"] = min(base_batch_size, 500_000)
        elif n_points > 5_000_000:  # Large
            optimized["batch_size"] = min(base_batch_size, 1_000_000)

        logger.debug(
            f"Optimized parameters: k={optimized['k_neighbors']}, "
            f"radius={optimized['search_radius']:.2f}, "
            f"batch_size={optimized['batch_size']:,}"
        )

        return optimized

    def _estimate_point_cloud_volume(self, points):
        """Estimate the volume of the point cloud bounding box."""
        if len(points) == 0:
            return 1.0

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords

        # Avoid zero volume
        dimensions = np.maximum(dimensions, 0.1)

        return np.prod(dimensions)

    def _should_cache_features(self, points, features):
        """Determine if features should be cached."""
        if not self._enable_feature_cache:
            return False

        # Estimate feature size
        feature_size = sum(arr.nbytes for arr in features.values()) / (
            1024 * 1024
        )  # MB

        # Don't cache if it would exceed max size
        if self._current_cache_size + feature_size > self._cache_max_size:
            return False

        # Cache if it's a reasonably sized result
        return 1 < feature_size < 50  # Cache features between 1MB and 50MB

    def _cache_features(self, cache_key, features):
        """Cache computed features."""
        feature_size = sum(arr.nbytes for arr in features.values()) / (
            1024 * 1024
        )  # MB

        # Clean cache if needed
        while (
            self._current_cache_size + feature_size > self._cache_max_size
            and self._feature_cache
        ):
            oldest_key = next(iter(self._feature_cache))
            old_features = self._feature_cache.pop(oldest_key)
            old_size = sum(arr.nbytes for arr in old_features.values()) / (1024 * 1024)
            self._current_cache_size -= old_size

        # Cache features
        self._feature_cache[cache_key] = features
        self._current_cache_size += feature_size

        logger.debug(
            f"Cached features ({feature_size:.1f}MB), total cache: {self._current_cache_size:.1f}MB"
        )

    def _update_performance_metrics(self, processing_time, num_points):
        """Update performance metrics."""
        if not self._enable_performance_metrics:
            return

        self._processing_times.append(processing_time)

        # Keep only recent metrics (last 100 computations)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)

        points_per_second = num_points / processing_time if processing_time > 0 else 0
        logger.debug(
            f"Performance: {processing_time:.2f}s, {points_per_second:.0f} points/sec"
        )

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()
        self._current_cache_size = 0
        logger.info("Feature cache cleared")

    def get_performance_summary(self):
        """Get performance summary."""
        if not hasattr(self, "_processing_times") or not self._processing_times:
            return {}

        return {
            "total_computations": len(self._processing_times),
            "avg_processing_time": np.mean(self._processing_times),
            "min_processing_time": np.min(self._processing_times),
            "max_processing_time": np.max(self._processing_times),
            "cache_hit_ratio": (
                len(self._feature_cache) / len(self._processing_times)
                if self._processing_times
                else 0
            ),
            "current_cache_size_mb": getattr(self, "_current_cache_size", 0),
            "strategy": self.strategy_name,
            "feature_mode": str(self.feature_mode),
            "adaptive_parameters": getattr(self, "_adaptive_parameters", {}).copy(),
        }

    def get_performance_insights(self, detailed: bool = False) -> dict:
        """
        Get comprehensive performance insights and recommendations.
        
        Provides actionable recommendations based on:
        - Cache hit ratios
        - GPU utilization (if available)
        - Processing time variance
        - Resource efficiency
        - Configuration optimization suggestions
        
        Args:
            detailed: If True, include detailed statistics and distributions
            
        Returns:
            Dictionary with insights, metrics, and recommendations
            
        Example:
            >>> insights = orchestrator.get_performance_insights()
            >>> print(insights['recommendations'])
            ['Enable caching to improve performance', ...]
            >>> if insights['gpu_available'] and not insights['gpu_utilized']:
            ...     print("Consider enabling GPU acceleration")
        """
        insights = {
            "timestamp": time.time(),
            "summary": {},
            "metrics": {},
            "recommendations": [],
            "warnings": [],
        }
        
        # Basic summary stats
        if not hasattr(self, "_processing_times") or not self._processing_times:
            insights["summary"]["status"] = "No data available"
            insights["recommendations"].append(
                "Process some tiles to collect performance metrics"
            )
            return insights
        
        # Processing time analysis
        times = np.array(self._processing_times)
        insights["summary"]["total_computations"] = len(times)
        insights["summary"]["avg_time_seconds"] = float(np.mean(times))
        insights["summary"]["total_time_seconds"] = float(np.sum(times))
        
        insights["metrics"]["processing_times"] = {
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "median": float(np.median(times)),
            "std": float(np.std(times)),
            "variance_coefficient": float(np.std(times) / np.mean(times)) if np.mean(times) > 0 else 0,
        }
        
        # Processing time variance analysis
        variance_coef = insights["metrics"]["processing_times"]["variance_coefficient"]
        if variance_coef > 0.5:
            insights["warnings"].append(
                f"High processing time variance (CV={variance_coef:.2f}). "
                "Tile sizes or point densities may be inconsistent."
            )
            insights["recommendations"].append(
                "Consider preprocessing tiles for consistent sizes"
            )
        
        # Cache analysis
        if hasattr(self, "_feature_cache") and hasattr(self, "_enable_feature_cache"):
            cache_enabled = self._enable_feature_cache
            cache_size = len(self._feature_cache)
            cache_memory_mb = getattr(self, "_current_cache_size", 0)
            cache_max_mb = getattr(self, "_cache_max_size_mb", 300)
            
            insights["metrics"]["cache"] = {
                "enabled": cache_enabled,
                "entries": cache_size,
                "size_mb": cache_memory_mb,
                "max_size_mb": cache_max_mb,
                "utilization_percent": (cache_memory_mb / cache_max_mb * 100) if cache_max_mb > 0 else 0,
            }
            
            # Cache hit ratio (estimate)
            # Note: Actual hits not tracked, estimating from cache size vs computations
            if cache_enabled and cache_size > 0:
                estimated_hit_ratio = min(cache_size / len(times), 1.0)
                insights["metrics"]["cache"]["estimated_hit_ratio"] = estimated_hit_ratio
                
                if estimated_hit_ratio < 0.1:
                    insights["recommendations"].append(
                        f"Cache hit ratio low (~{estimated_hit_ratio:.0%}). "
                        "Consider increasing cache size or reviewing feature computation patterns."
                    )
                elif estimated_hit_ratio > 0.7:
                    insights["summary"]["cache_status"] = "✅ Excellent cache utilization"
            else:
                if not cache_enabled:
                    insights["recommendations"].append(
                        "Enable feature caching to improve performance for repeated tiles"
                    )
                insights["summary"]["cache_status"] = "⚠️  Cache disabled or empty"
        
        # GPU utilization analysis
        if hasattr(self, "gpu_available") and hasattr(self, "use_gpu"):
            insights["metrics"]["gpu"] = {
                "available": self.gpu_available,
                "requested": getattr(self, "use_gpu", False),
                "strategy": self.strategy_name,
            }
            
            if self.gpu_available:
                # Try to get GPU memory info
                try:
                    from ..core.error_handler import get_gpu_memory_info
                    gpu_mem = get_gpu_memory_info()
                    if gpu_mem:
                        insights["metrics"]["gpu"]["memory_gb"] = {
                            "used": gpu_mem.get("used_gb", 0),
                            "total": gpu_mem.get("device_total_gb", 0),
                            "free": gpu_mem.get("free_gb", 0),
                            "utilization_percent": (
                                gpu_mem.get("used_gb", 0) / gpu_mem.get("device_total_gb", 1) * 100
                            )
                        }
                        
                        # GPU memory recommendations
                        gpu_util = insights["metrics"]["gpu"]["memory_gb"]["utilization_percent"]
                        if gpu_util < 20:
                            insights["recommendations"].append(
                                f"GPU memory underutilized ({gpu_util:.0f}%). "
                                "Consider increasing chunk size for better GPU efficiency."
                            )
                        elif gpu_util > 90:
                            insights["warnings"].append(
                                f"GPU memory heavily utilized ({gpu_util:.0f}%). "
                                "Risk of OOM errors."
                            )
                            insights["recommendations"].append(
                                "Reduce chunk size or use GPU chunked mode"
                            )
                except Exception as e:
                    logger.debug(f"Could not get GPU memory info: {e}")
                
                # GPU utilization recommendations
                if self.gpu_available and not getattr(self, "use_gpu", False):
                    insights["recommendations"].append(
                        "🚀 GPU available but not utilized. Enable GPU to accelerate processing (3-10x faster)."
                    )
                    insights["summary"]["gpu_status"] = "⚠️  GPU available but unused"
                elif self.gpu_available and getattr(self, "use_gpu", False):
                    insights["summary"]["gpu_status"] = "✅ GPU acceleration enabled"
            else:
                if getattr(self, "use_gpu", False):
                    insights["warnings"].append(
                        "GPU requested but not available. Using CPU fallback."
                    )
                insights["summary"]["gpu_status"] = "💻 CPU mode"
        
        # Strategy analysis
        insights["metrics"]["strategy"] = {
            "name": self.strategy_name,
            "feature_mode": str(self.feature_mode),
        }
        
        # Multi-scale analysis
        if hasattr(self, "use_multi_scale") and self.use_multi_scale:
            insights["metrics"]["multi_scale"] = {
                "enabled": True,
                "gpu_accelerated": getattr(getattr(self, "multi_scale_computer", None), "use_gpu", False)
            }
            insights["summary"]["multi_scale_status"] = "✅ Multi-scale enabled"
        
        # RGB/NIR analysis
        rgb_enabled = getattr(self, "use_rgb", False)
        nir_enabled = getattr(self, "use_infrared", False)
        insights["metrics"]["spectral_data"] = {
            "rgb_enabled": rgb_enabled,
            "nir_enabled": nir_enabled,
            "rgb_fetcher_available": hasattr(self, "rgb_fetcher") and self.rgb_fetcher is not None,
            "nir_fetcher_available": hasattr(self, "infrared_fetcher") and self.infrared_fetcher is not None,
        }
        
        # Detailed statistics if requested
        if detailed:
            insights["detailed"] = {
                "processing_time_percentiles": {
                    "p10": float(np.percentile(times, 10)),
                    "p25": float(np.percentile(times, 25)),
                    "p50": float(np.percentile(times, 50)),
                    "p75": float(np.percentile(times, 75)),
                    "p90": float(np.percentile(times, 90)),
                    "p95": float(np.percentile(times, 95)),
                    "p99": float(np.percentile(times, 99)),
                },
                "processing_time_histogram": np.histogram(times, bins=10)[0].tolist(),
            }
            
            # Adaptive parameters if available
            if hasattr(self, "_adaptive_parameters"):
                insights["detailed"]["adaptive_parameters"] = getattr(
                    self, "_adaptive_parameters", {}
                ).copy()
        
        # Overall recommendations
        avg_time = insights["summary"]["avg_time_seconds"]
        if avg_time > 10:
            insights["recommendations"].append(
                f"Average processing time is {avg_time:.1f}s. "
                "Consider enabling GPU or optimizing parameters."
            )
        elif avg_time < 1:
            insights["summary"]["performance_status"] = "✅ Excellent performance"
        
        # Configuration recommendations
        if not insights["recommendations"]:
            insights["recommendations"].append(
                "✅ Performance is optimized. No specific recommendations."
            )
        
        return insights

    def print_performance_insights(self, detailed: bool = False):
        """
        Print formatted performance insights to console.
        
        Args:
            detailed: If True, include detailed statistics
            
        Example:
            >>> orchestrator.print_performance_insights()
            
            ╔══════════════════════════════════════════════════════════════╗
            ║         PERFORMANCE INSIGHTS - Feature Orchestrator         ║
            ╚══════════════════════════════════════════════════════════════╝
            
            📊 SUMMARY
            ─────────────────────────────────────────────────────────────
            Total Computations: 42
            Average Time: 3.45s
            GPU Status: ✅ GPU acceleration enabled
            ...
        """
        insights = self.get_performance_insights(detailed=detailed)
        
        # Header
        print("\n" + "╔" + "═" * 62 + "╗")
        print("║" + " " * 10 + "PERFORMANCE INSIGHTS - Feature Orchestrator" + " " * 9 + "║")
        print("╚" + "═" * 62 + "╝\n")
        
        # Summary section
        if insights.get("summary"):
            print("📊 SUMMARY")
            print("─" * 64)
            summary = insights["summary"]
            if "total_computations" in summary:
                print(f"Total Computations: {summary['total_computations']}")
            if "avg_time_seconds" in summary:
                print(f"Average Time: {summary['avg_time_seconds']:.2f}s")
            if "total_time_seconds" in summary:
                print(f"Total Time: {summary['total_time_seconds']:.1f}s")
            if "gpu_status" in summary:
                print(f"GPU Status: {summary['gpu_status']}")
            if "cache_status" in summary:
                print(f"Cache Status: {summary['cache_status']}")
            if "multi_scale_status" in summary:
                print(f"Multi-Scale: {summary['multi_scale_status']}")
            if "performance_status" in summary:
                print(f"Performance: {summary['performance_status']}")
            print()
        
        # Metrics section
        if insights.get("metrics"):
            print("📈 METRICS")
            print("─" * 64)
            
            # Processing times
            if "processing_times" in insights["metrics"]:
                pt = insights["metrics"]["processing_times"]
                print(f"  Processing Time (s):")
                print(f"    Min: {pt['min']:.2f}  |  Max: {pt['max']:.2f}  |  Median: {pt['median']:.2f}")
                print(f"    Std Dev: {pt['std']:.2f}  |  Variance Coef: {pt['variance_coefficient']:.2f}")
            
            # Cache metrics
            if "cache" in insights["metrics"]:
                cache = insights["metrics"]["cache"]
                print(f"  Cache:")
                print(f"    Enabled: {cache['enabled']}  |  Entries: {cache['entries']}")
                print(f"    Memory: {cache['size_mb']:.1f}/{cache['max_size_mb']:.0f} MB ({cache['utilization_percent']:.0f}%)")
                if "estimated_hit_ratio" in cache:
                    print(f"    Est. Hit Ratio: {cache['estimated_hit_ratio']:.1%}")
            
            # GPU metrics
            if "gpu" in insights["metrics"]:
                gpu = insights["metrics"]["gpu"]
                print(f"  GPU:")
                print(f"    Available: {gpu['available']}  |  Requested: {gpu['requested']}")
                print(f"    Strategy: {gpu['strategy']}")
                if "memory_gb" in gpu:
                    mem = gpu["memory_gb"]
                    print(f"    Memory: {mem['used']:.1f}/{mem['total']:.1f} GB ({mem['utilization_percent']:.0f}%)")
            print()
        
        # Warnings
        if insights.get("warnings"):
            print("⚠️  WARNINGS")
            print("─" * 64)
            for i, warning in enumerate(insights["warnings"], 1):
                print(f"  {i}. {warning}")
            print()
        
        # Recommendations
        if insights.get("recommendations"):
            print("💡 RECOMMENDATIONS")
            print("─" * 64)
            for i, rec in enumerate(insights["recommendations"], 1):
                print(f"  {i}. {rec}")
            print()
        
        # Detailed statistics
        if detailed and "detailed" in insights:
            print("🔍 DETAILED STATISTICS")
            print("─" * 64)
            det = insights["detailed"]
            if "processing_time_percentiles" in det:
                perc = det["processing_time_percentiles"]
                print("  Processing Time Percentiles (s):")
                print(f"    P10: {perc['p10']:.2f}  P25: {perc['p25']:.2f}  P50: {perc['p50']:.2f}")
                print(f"    P75: {perc['p75']:.2f}  P90: {perc['p90']:.2f}  P95: {perc['p95']:.2f}")
            print()
        
        print("─" * 64)

    def _start_parallel_rgb_nir_processing(self, tile_data):
        """Start parallel RGB/NIR processing."""

        def fetch_rgb_nir():
            """Fetch RGB/NIR data in parallel."""
            results = {}
            points = tile_data["points"]

            if self.use_rgb and self.rgb_fetcher:
                try:
                    rgb_data = self.rgb_fetcher.fetch_for_points(points)
                    results["rgb"] = rgb_data
                except (ConnectionError, TimeoutError) as e:
                    error = DataFetchError.create(
                        data_type="RGB orthophoto",
                        error=e
                    )
                    logger.warning(str(error))
                except (OSError, IOError) as e:
                    logger.warning(f"RGB fetch I/O error: {e}")
                    logger.warning("  Check cache directory permissions and disk space")
                except Exception as e:
                    error = DataFetchError.create(
                        data_type="RGB orthophoto",
                        error=e
                    )
                    logger.warning(str(error))

            if self.use_infrared and self.infrared_fetcher:
                try:
                    nir_data = self.infrared_fetcher.fetch_for_points(points)
                    results["nir"] = nir_data
                except (ConnectionError, TimeoutError) as e:
                    error = DataFetchError.create(
                        data_type="NIR infrared",
                        error=e
                    )
                    logger.warning(str(error))
                except (OSError, IOError) as e:
                    logger.warning(f"NIR fetch I/O error: {e}")
                    logger.warning("  Check cache directory permissions and disk space")
                except Exception as e:
                    error = DataFetchError.create(
                        data_type="NIR infrared",
                        error=e
                    )
                    logger.warning(str(error))

            return results

        return self._rgb_nir_executor.submit(fetch_rgb_nir)

    def _compute_geometric_features_optimized(
        self, points, classification, optimized_params, **kwargs
    ):
        """
        Compute geometric features with optimized parameters.

        Note: This optimization only works with Strategy Pattern computers that have
        k_neighbors attribute. FeatureComputer uses k values passed to methods.
        """
        processor_cfg = self.config.get("processor", {})
        use_feature_computer = processor_cfg.get("use_feature_computer", False)

        # Only apply k_neighbors optimization for Strategy Pattern
        # FeatureComputer takes k as method parameter, not attribute
        if (
            not use_feature_computer
            and optimized_params
            and hasattr(self.computer, "k_neighbors")
        ):
            original_k = getattr(self.computer, "k_neighbors", 20)
            setattr(
                self.computer,
                "k_neighbors",
                optimized_params.get("k_neighbors", original_k),
            )

            try:
                # Use the existing _compute_geometric_features method
                return self._compute_geometric_features(
                    points, classification, **kwargs
                )
            finally:
                # Restore original parameters
                setattr(self.computer, "k_neighbors", original_k)
        else:
            # No optimization to apply (no optimized params)
            return self._compute_geometric_features(points, classification, **kwargs)

    def validate_mode(self, mode: FeatureMode) -> bool:
        """
        Validate that a feature mode is supported.

        Args:
            mode: FeatureMode to validate

        Returns:
            bool: True if mode is valid and supported
        """
        return isinstance(mode, FeatureMode)

    def get_feature_list(self, mode: Optional[FeatureMode] = None) -> List[str]:
        """
        Get list of features for a given mode.

        Args:
            mode: FeatureMode to query (defaults to current mode)

        Returns:
            List of feature names that will be computed
        """
        if mode is None:
            mode = self.feature_mode

        mode_str = mode.value if isinstance(mode, FeatureMode) else mode
        feature_config = get_feature_config(mode_str)
        return list(feature_config.feature_names)

    def filter_features(
        self, features: Dict[str, np.ndarray], mode: Optional[FeatureMode] = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter features dict to only include features for given mode.

        This is used to enforce feature mode consistency across all
        computation strategies.

        Args:
            features: Dict of all computed features
            mode: FeatureMode to filter by (defaults to current mode)

        Returns:
            Filtered dict with only features for specified mode
        """
        if mode is None:
            mode = self.feature_mode

        # Get allowed features for this mode
        allowed_features = set(self.get_feature_list(mode))

        # Always keep core features regardless of mode
        core_features = {
            "normals",
            "curvature",
            "height",
            "intensity",
            "return_number",
            "is_ground",
            "is_synthetic",
        }
        allowed_features.update(core_features)

        # Always keep cluster ID features if they were computed
        # These are ground-truth-based spatial features
        cluster_features = {
            "building_cluster_id",
            "parcel_cluster_id",
        }
        allowed_features.update(cluster_features)

        # Handle spectral features: if mode defines 'red', 'green', 'blue' individually,
        # also allow 'rgb' as a combined feature (and vice versa)
        if (
            "red" in allowed_features
            or "green" in allowed_features
            or "blue" in allowed_features
        ):
            allowed_features.add("rgb")
        if "rgb" in allowed_features:
            allowed_features.update(["red", "green", "blue"])

        # Same for NIR and NDVI
        if "nir" in allowed_features or self.use_infrared:
            allowed_features.add("nir")
        if "ndvi" in allowed_features or (self.use_rgb and self.use_infrared):
            allowed_features.add("ndvi")

        # Filter
        filtered = {
            k: v
            for k, v in features.items()
            if k in allowed_features or k.startswith("enriched_")
        }

        return filtered

    # =========================================================================
    # COMPUTATION COORDINATION (from FeatureComputer)
    # =========================================================================

    def compute_features(
        self, tile_data: Dict[str, Any], use_enriched: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a point cloud tile with V5 optimizations.

        This version includes:
        1. Intelligent caching with memory management
        2. Automatic parameter optimization based on data characteristics
        3. Parallel RGB/NIR processing
        4. Performance monitoring and adaptive tuning
        5. All original feature computation capabilities
        6. Spatial artifact filtering (v3.1.0) for planarity, linearity, horizontality

        Args:
            tile_data: Dictionary containing:
                - points: (N, 3) XYZ coordinates
                - classification: (N,) classification codes
                - intensity: (N,) intensity values
                - return_number: (N,) return numbers
                - input_rgb: Optional (N, 3) RGB from input LAZ
                - input_nir: Optional (N,) NIR from input LAZ
                - input_ndvi: Optional (N,) NDVI from input LAZ
                - enriched_features: Optional dict of enriched features
            use_enriched: If True and enriched features exist, use them

        Returns:
            Dictionary of computed features with optimizations applied
        """
        import time

        start_time = time.time()

        points = tile_data["points"]
        classification = tile_data["classification"]
        intensity = tile_data["intensity"]
        return_number = tile_data["return_number"]
        enriched_features = tile_data.get("enriched_features", {})

        # V5 OPTIMIZATION: Check cache first
        cache_key = None
        if hasattr(self, "_enable_feature_cache") and self._enable_feature_cache:
            cache_key = self._generate_cache_key(
                points,
                classification,
                {"use_enriched": use_enriched, "mode": str(self.feature_mode)},
            )
            if cache_key in self._feature_cache:
                logger.debug("Features retrieved from cache")
                return self._feature_cache[cache_key]

        # V5 OPTIMIZATION: Optimize parameters based on data characteristics
        optimized_params = {}
        if hasattr(self, "_adaptive_parameters"):
            optimized_params = self._optimize_parameters_for_data(
                points, classification
            )

        all_features = {}

        # Check if we should use existing enriched features
        if use_enriched and enriched_features:
            logger.info("  ♻️  Using existing enriched features from input LAZ")

            normals = enriched_features.get("normals")
            curvature = enriched_features.get("curvature")
            # Use explicit None check to avoid numpy array truthiness issues
            height = enriched_features.get("height")
            if height is None:
                height = enriched_features.get("z_normalized")

            # Build geo_features from enriched
            geo_features = {
                k: v
                for k, v in enriched_features.items()
                if k not in ["normals", "curvature", "height"]
            }
        else:
            # Check data availability for better logging
            has_rgb = tile_data.get("input_rgb") is not None or (
                self.rgb_fetcher is not None
            )
            has_nir = tile_data.get("input_nir") is not None or (
                self.infrared_fetcher is not None
            )

            # V5 OPTIMIZATION: Start parallel RGB/NIR processing if available
            rgb_nir_future = None
            if hasattr(self, "_rgb_nir_executor") and (
                self.use_rgb or self.use_infrared
            ):
                rgb_nir_future = self._start_parallel_rgb_nir_processing(tile_data)

            # Compute features with optimized parameters
            normals, curvature, height, geo_features = (
                self._compute_geometric_features_optimized(
                    points,
                    classification,
                    optimized_params,
                    has_rgb=has_rgb,
                    has_nir=has_nir,
                )
            )

            # V5 OPTIMIZATION: Integrate parallel RGB/NIR results
            if rgb_nir_future is not None:
                try:
                    parallel_rgb_nir = rgb_nir_future.result(timeout=30)
                    # Merge parallel results into tile_data for later processing
                    if "rgb" in parallel_rgb_nir:
                        tile_data["fetched_rgb"] = parallel_rgb_nir["rgb"]
                    if "nir" in parallel_rgb_nir:
                        tile_data["fetched_nir"] = parallel_rgb_nir["nir"]
                except TimeoutError as e:
                    logger.warning("Parallel RGB/NIR processing timed out after 30s")
                    logger.warning("  Consider increasing timeout or checking network")
                except (ConnectionError, OSError) as e:
                    error = DataFetchError.create(
                        data_type="RGB/NIR parallel fetch",
                        error=e
                    )
                    logger.warning(str(error))
                except Exception as e:
                    logger.warning(f"Parallel RGB/NIR processing failed: {e}")
                    logger.debug(f"  Error type: {type(e).__name__}")

        # Add main features
        all_features["normals"] = normals
        all_features["curvature"] = curvature
        all_features["height"] = height
        all_features["intensity"] = intensity
        all_features["return_number"] = return_number

        # Add geometric features
        if isinstance(geo_features, dict):
            all_features.update(geo_features)

        # Add enriched features if present (alongside recomputed ones)
        if enriched_features:
            for feat_name, feat_data in enriched_features.items():
                # Use prefix to distinguish from recomputed features
                enriched_key = (
                    f"enriched_{feat_name}" if feat_name in all_features else feat_name
                )
                all_features[enriched_key] = feat_data
            logger.info(
                f"  ✓ Added {len(enriched_features)} enriched features from input"
            )

        # Add spectral features (RGB, NIR, NDVI)
        rgb_added = self._add_rgb_features(tile_data, all_features)
        nir_added = self._add_nir_features(tile_data, all_features)
        self._add_ndvi_features(tile_data, all_features, rgb_added, nir_added)

        # Add is_ground feature (with DTM augmentation support)
        self._add_is_ground_feature(tile_data, all_features)

        # Add architectural style if requested
        self._add_architectural_style(tile_data, all_features)

        # Add cluster ID features if requested (optional spatial clustering)
        self._add_cluster_id_features(tile_data, all_features)

        # Add plane features if requested (architectural plane detection)
        self._add_plane_features(tile_data, all_features)

        # Add building-plane features if requested (hierarchical clustering)
        self._add_building_plane_features(tile_data, all_features)

        # Enforce feature mode (filter to only allowed features)
        if self.feature_mode != FeatureMode.FULL:
            all_features = self.filter_features(all_features, self.feature_mode)
            logger.debug(
                f"  🔽 Filtered to {len(all_features)} features for mode {self.feature_mode.value}"
            )

        # ✅ PHASE 2: Validate features before use
        from .validation import validate_features, check_feature_sanity

        if not check_feature_sanity(all_features, points):
            logger.error("Feature sanity check failed!")

        # Fix invalid values (NaN, Inf, out-of-range)
        all_features, issue_counts = validate_features(
            all_features, fix_invalid=True, verbose=True
        )

        if issue_counts:
            logger.info(
                f"  ✅ Validated features: fixed {sum(issue_counts.values())} "
                f"invalid values in {len(issue_counts)} features"
            )

        # 🆕 PHASE 3: Apply spatial artifact filtering (v3.1.0)
        # Remove scan line artifacts from planarity, linearity, horizontality
        features_cfg = self.config.get("features", {})
        enable_filtering = features_cfg.get("enable_artifact_filtering", True)

        if enable_filtering:
            from .compute.feature_filter import smooth_feature_spatial

            filter_start = time.time()
            filtered_count = 0

            # Features that need artifact filtering
            artifact_prone_features = ["planarity", "linearity", "horizontality"]

            # v3.1.1: Get threshold from config or use default
            filter_threshold = features_cfg.get("artifact_filter_threshold", 0.2)

            for feature_name in artifact_prone_features:
                if feature_name in all_features:
                    original = all_features[feature_name]

                    # Apply spatial filtering (v3.1.1: deviation-based algorithm)
                    filtered = smooth_feature_spatial(
                        feature=original,
                        points=points,
                        k_neighbors=15,
                        std_threshold=filter_threshold,
                        feature_name=feature_name,
                    )

                    # Count how many values changed
                    changes = np.sum(np.abs(filtered - original) > 1e-6)
                    if changes > 0:
                        all_features[feature_name] = filtered
                        filtered_count += 1
                        logger.debug(
                            f"  🔧 Filtered {feature_name}: {changes} points modified"
                        )

            if filtered_count > 0:
                filter_time = time.time() - filter_start
                logger.info(
                    f"  🧹 Applied artifact filtering to {filtered_count} features "
                    f"in {filter_time:.2f}s"
                )

        # Public feature-name alias: SUPPORTED_FEAT_KEYS (ptv3_formatter.py) and
        # the feature_modes.py mode definitions all name this feature
        # "height_above_ground", but this pipeline has always stored it
        # internally under "height" (see core_features in filter_features()).
        # Without this alias, Ptv3Formatter._build_feat() never finds
        # "height_above_ground" in the patch dict and silently zero-fills the
        # whole column for every point/patch — height above ground is a core
        # building-vs-vegetation-vs-ground cue, so this was a critical bug for
        # any PTv3/Pointcept export.
        if "height" in all_features and "height_above_ground" not in all_features:
            all_features["height_above_ground"] = all_features["height"]

        # Log feature quality metrics for debugging
        self._log_feature_quality(all_features)

        # V5 OPTIMIZATION: Cache results and update performance metrics
        processing_time = time.time() - start_time

        if (
            hasattr(self, "_enable_feature_cache")
            and cache_key
            and self._should_cache_features(points, all_features)
        ):
            self._cache_features(cache_key, all_features)

        if (
            hasattr(self, "_enable_performance_metrics")
            and self._enable_performance_metrics
        ):
            self._update_performance_metrics(processing_time, len(points))

        return all_features

    def _log_feature_quality(self, features: Dict[str, np.ndarray]) -> None:
        """
        Log feature quality metrics for debugging artifacts.

        Helps identify numerical instability issues in geometric features.
        """
        quality_report = []

        # Check critical geometric features
        for name in ["planarity", "linearity", "horizontality", "verticality"]:
            if name not in features:
                continue

            values = features[name]
            n_total = len(values)

            # Count edge values (potential artifacts after clamping)
            n_zero = np.sum(values == 0.0)
            n_one = np.sum(values == 1.0)
            n_invalid = np.sum(~np.isfinite(values))

            # Only report if significant issues detected
            pct_zero = 100 * n_zero / n_total
            pct_one = 100 * n_one / n_total

            if n_invalid > 0 or pct_zero > 5.0 or pct_one > 5.0:
                quality_report.append(
                    f"{name}: {n_zero} at 0.0 ({pct_zero:.1f}%), "
                    f"{n_one} at 1.0 ({pct_one:.1f}%), "
                    f"{n_invalid} invalid"
                )

        if quality_report:
            logger.info("  📊 Feature Quality Check:")
            for line in quality_report:
                logger.info(f"      {line}")

    def _compute_geometric_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        has_rgb: bool = False,
        has_nir: bool = False,
    ) -> tuple:
        """
        Compute geometric features using selected strategy.

        Args:
            points: (N, 3) XYZ coordinates
            classification: (N,) classification codes
            has_rgb: Whether RGB data is available
            has_nir: Whether NIR data is available

        Returns:
            Tuple of (normals, curvature, height, geo_features_dict)
        """
        features_cfg = self.config.get("features", {})
        processor_cfg = self.config.get("processor", {})

        k_neighbors = features_cfg.get("k_neighbors")
        search_radius = features_cfg.get("search_radius", None)

        # Derive include_extra from feature mode (not from config)
        # Only MINIMAL mode should exclude extra features
        include_extra = self.feature_mode != FeatureMode.MINIMAL

        k_display = k_neighbors if k_neighbors else "auto"

        # Log search strategy
        if search_radius is not None and search_radius > 0:
            logger.info(
                f"  🔧 Computing features | radius={search_radius:.2f}m (avoids scan line artifacts) | mode={self.feature_mode.value}"
            )
        else:
            logger.info(
                f"  🔧 Computing features | k={k_display} | mode={self.feature_mode.value}"
            )

        feature_start = time.time()

        # Compute patch center for distance_to_center feature
        patch_center = np.mean(points, axis=0) if include_extra else None

        # Use manual k if specified, otherwise auto-estimate
        use_auto_k = k_neighbors is None
        k_value = k_neighbors if k_neighbors is not None else 20

        # NEW (v6.2): Multi-scale feature computation
        if self.use_multi_scale and self.multi_scale_computer is not None:
            logger.debug("Using multi-scale feature computation")

            # Determine which features to compute
            if include_extra:
                feature_list = [
                    "planarity",
                    "linearity",
                    "sphericity",
                    "verticality",
                    "horizontality",
                ]
            else:
                # Minimal mode - just normals and curvature
                feature_list = ["planarity"]

            # Compute multi-scale features
            try:
                multi_scale_features = self.multi_scale_computer.compute_features(
                    points=points, features_to_compute=feature_list
                )

                # Extract normals and curvature
                # (multi-scale returns these as by-products)
                normals = multi_scale_features.get(
                    "normals", np.zeros((len(points), 3))
                )
                curvature = multi_scale_features.get("curvature", np.zeros(len(points)))

                # Compute height
                z_min = np.min(points[:, 2])
                height = points[:, 2] - z_min

                # Return multi-scale features as geo_features
                geo_features = {
                    k: v
                    for k, v in multi_scale_features.items()
                    if k not in ["normals", "curvature"]
                }

                # Add distance_to_center if needed
                if include_extra and patch_center is not None:
                    distances = np.linalg.norm(points - patch_center, axis=1)
                    geo_features["distance_to_center"] = distances

                logger.info(
                    f"  ✓ Multi-scale features computed | "
                    f"{len(geo_features)} features | "
                    f"time={time.time()-feature_start:.2f}s"
                )

                return normals, curvature, height, geo_features

            except (ValueError, IndexError) as e:
                logger.error(
                    f"Multi-scale computation failed with invalid data: {e}. "
                    f"Falling back to standard computation."
                )
                # Fall through to standard computation
            except ImportError as e:
                error = InitializationError.create(
                    component="multi-scale computer",
                    error=e
                )
                logger.error(str(error))
                logger.error("Falling back to standard computation.")
                # Fall through to standard computation
            except Exception as e:
                error = FeatureComputationError.create(
                    feature_name="multi-scale features",
                    error=e,
                    num_points=len(points)
                )
                logger.error(str(error))
                logger.error("Falling back to standard computation.")
                # Fall through to standard computation

        # NEW (Phase 4 Task 1.4): Check which computer API to use
        processor_cfg = self.config.get("processor", {})
        use_feature_computer = processor_cfg.get("use_feature_computer", False)

        if use_feature_computer:
            # Use FeatureComputer API (Phase 4)
            logger.debug("Using FeatureComputer.compute_all_features()")

            # Map geometric features based on include_extra
            if include_extra:
                geometric_features = [
                    "planarity",
                    "linearity",
                    "sphericity",
                    "anisotropy",
                    "eigenentropy",
                    "omnivariance",
                    "verticality",
                ]
            else:
                geometric_features = []  # Minimal mode

            # Call API
            feature_dict = self.computer.compute_all_features(
                points=points,
                k_normals=k_value,
                k_curvature=k_value,
                k_geometric=k_value,
                geometric_features=geometric_features,
                mode=None,  # Use automatic mode selection
            )

            # Add height (z-normalized) if not present
            if "height" not in feature_dict:
                z_min = np.min(points[:, 2])
                feature_dict["height"] = points[:, 2] - z_min

            # Add distance_to_center if needed and not present
            if (
                include_extra
                and "distance_to_center" not in feature_dict
                and patch_center is not None
            ):
                distances = np.linalg.norm(points - patch_center, axis=1)
                feature_dict["distance_to_center"] = distances

        else:
            # Use Strategy Pattern API (legacy)
            logger.debug("Using Strategy.compute_features()")

            # ✅ OPTIMIZATION (v3.5.3): Check cache for intermediate results
            # This can save 15-25% of computation time when features need normals/eigenvalues
            cached_intermediates = None
            if hasattr(self, '_get_cached_normals_eigenvalues'):
                cached_intermediates = self._get_cached_normals_eigenvalues(points, k_value)
            
            # Pass cached intermediates to strategy (if it supports caching)
            compute_kwargs = {
                "points": points,
                "classification": classification,
                "auto_k": use_auto_k,
                "include_extra": include_extra,
                "patch_center": patch_center,
                "mode": self.feature_mode.value,
                "radius": search_radius,
            }
            
            # Add cached intermediates if available
            if cached_intermediates is not None and hasattr(self.computer, 'set_cached_intermediates'):
                self.computer.set_cached_intermediates(cached_intermediates)
            
            # Compute features using selected computer
            feature_dict = self.computer.compute_features(**compute_kwargs)
            
            # ✅ OPTIMIZATION (v3.5.3): Cache normals/eigenvalues for reuse
            # If strategy computed them, cache for next time
            if (cached_intermediates is None  # Wasn't cached
                and hasattr(self, '_cache_normals_eigenvalues')
                and 'normals' in feature_dict):
                
                # Check if eigenvalues are available (some strategies compute them)
                eigenvalues = feature_dict.get('eigenvalues')
                if eigenvalues is not None:
                    self._cache_normals_eigenvalues(
                        points, k_value,
                        feature_dict['normals'],
                        eigenvalues
                    )
                    logger.debug(f"✅ Cached normals/eigenvalues for k={k_value}")

        # Extract main features
        normals = feature_dict.get("normals")
        curvature = feature_dict.get("curvature")
        height = feature_dict.get("height")

        # Extract geometric features
        main_features = {"normals", "curvature", "height"}
        geo_features = {k: v for k, v in feature_dict.items() if k not in main_features}

        elapsed = time.time() - feature_start
        logger.info(
            f"  ✓ Computed {len(geo_features)} geometric features "
            f"in {elapsed:.2f}s using {self.strategy_name}"
        )

        return normals, curvature, height, geo_features

    # =========================================================================
    # SPECTRAL FEATURES (RGB, NIR, NDVI)
    # =========================================================================

    def _add_rgb_features(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add RGB features from input LAZ or fetch from orthophotos.

        Args:
            tile_data: Tile data dict
            all_features: Features dict to update

        Returns:
            bool: True if RGB was added
        """
        if not self.use_rgb:
            return False

        # Check if RGB already in input
        input_rgb = tile_data.get("input_rgb")
        if input_rgb is not None:
            all_features["rgb"] = input_rgb
            logger.info("  ✓ Using RGB from input LAZ")
            return True

        # Try to fetch from orthophotos
        if self.rgb_fetcher is not None:
            try:
                points = tile_data["points"]
                rgb = self.rgb_fetcher.augment_points_with_rgb(points)
                if rgb is not None:
                    # Normalize RGB using GPU-accelerated utility
                    use_gpu = self.use_gpu and is_gpu_available()
                    all_features["rgb"] = normalize_rgb(rgb, use_gpu=use_gpu)
                    logger.info(f"  ✓ Fetched RGB from IGN orthophotos (GPU: {use_gpu})")
                    return True
            except (ConnectionError, TimeoutError) as e:
                error = DataFetchError.create(
                    data_type="RGB orthophoto",
                    error=e
                )
                logger.warning(str(error))
            except (OSError, IOError) as e:
                logger.warning(f"  ⚠️  RGB fetch I/O error: {e}")
                logger.warning("  Check cache directory and disk space")
            except (ValueError, IndexError) as e:
                logger.warning(f"  ⚠️  Invalid RGB data: {e}")
            except Exception as e:
                error = DataFetchError.create(
                    data_type="RGB orthophoto",
                    error=e
                )
                logger.warning(str(error))

        return False

    def _add_nir_features(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ) -> bool:
        """
        Add NIR (near-infrared) features from input LAZ or fetch.

        Args:
            tile_data: Tile data dict
            all_features: Features dict to update

        Returns:
            bool: True if NIR was added
        """
        if not self.use_infrared:
            return False

        # Check if NIR already in input
        input_nir = tile_data.get("input_nir")
        if input_nir is not None:
            all_features["nir"] = input_nir
            logger.info("  ✓ Using NIR from input LAZ")
            return True

        # Try to fetch from infrared service
        if self.infrared_fetcher is not None:
            try:
                points = tile_data["points"]
                nir = self.infrared_fetcher.augment_points_with_infrared(points)
                if nir is not None:
                    # Normalize NIR using GPU-accelerated utility
                    use_gpu = self.use_gpu and is_gpu_available()
                    all_features["nir"] = normalize_nir(nir, use_gpu=use_gpu)
                    logger.info(f"  ✓ Fetched NIR from IGN IRC (GPU: {use_gpu})")
                    return True
            except (ConnectionError, TimeoutError) as e:
                error = DataFetchError.create(
                    data_type="NIR infrared",
                    error=e
                )
                logger.warning(str(error))
            except (OSError, IOError) as e:
                logger.warning(f"  ⚠️  NIR fetch I/O error: {e}")
                logger.warning("  Check cache directory and disk space")
            except (ValueError, IndexError) as e:
                logger.warning(f"  ⚠️  Invalid NIR data: {e}")
            except Exception as e:
                error = DataFetchError.create(
                    data_type="NIR infrared",
                    error=e
                )
                logger.warning(str(error))

        return False

    def _add_ndvi_features(
        self,
        tile_data: Dict[str, Any],
        all_features: Dict[str, np.ndarray],
        rgb_added: bool,
        nir_added: bool,
    ):
        """
        Add or compute NDVI (Normalized Difference Vegetation Index).

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
            rgb_added: Whether RGB was successfully added
            nir_added: Whether NIR was successfully added
        """
        # Check both processor and features sections for compute_ndvi flag
        processor_cfg = self.config.get("processor", {})
        features_cfg = self.config.get("features", {})
        compute_ndvi = features_cfg.get(
            "compute_ndvi", processor_cfg.get("compute_ndvi", False)
        )

        if not compute_ndvi:
            return

        # Check if NDVI already in input
        input_ndvi = tile_data.get("input_ndvi")
        if input_ndvi is not None:
            all_features["ndvi"] = input_ndvi
            logger.info("  ✓ Using NDVI from input LAZ")
            return

        # Compute NDVI if we have both RGB and NIR
        if rgb_added and nir_added:
            rgb = all_features["rgb"]
            nir = all_features["nir"]

            # Extract red channel (index 0)
            red = rgb[:, 0].astype(np.float32)
            nir_float = nir.astype(np.float32)

            # Compute NDVI with safe division
            denominator = nir_float + red
            ndvi = np.zeros_like(nir_float)
            mask = denominator > 0
            ndvi[mask] = (nir_float[mask] - red[mask]) / denominator[mask]

            # Validate NDVI: clip to valid range [-1, 1] and fix any NaN/Inf
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

            all_features["ndvi"] = ndvi
            logger.info("  ✓ Computed NDVI from RGB and NIR")
        else:
            # Don't add NDVI to features if we can't compute it
            # This prevents None/empty values from being saved
            logger.debug(
                "  ⚠️  Cannot compute NDVI (need both RGB and NIR) - not adding to features"
            )

    def _add_is_ground_feature(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ):
        """
        Add is_ground binary feature with DTM augmentation support.

        Computes a binary indicator (0/1) for ground points, with optional
        support for detecting DTM-augmented synthetic ground points.

        Args:
            tile_data: Tile data dict (must contain 'classification')
            all_features: Features dict to update

        Notes:
            - Ground points are ASPRS class 2
            - Synthetic points from DTM augmentation can be included/excluded
            - Logs statistics about ground coverage and DTM contribution
        """
        features_cfg = self.config.get("features", {})
        processor_cfg = self.config.get("processor", {})

        # Check if is_ground feature is requested
        compute_is_ground = features_cfg.get(
            "compute_is_ground", processor_cfg.get("compute_is_ground", True)
        )

        if not compute_is_ground:
            return

        try:
            from .compute.is_ground import compute_is_ground_with_stats

            classification = tile_data["classification"]

            # Check for synthetic flags from DTM augmentation
            synthetic_flags = tile_data.get("synthetic_flags")

            # Check whether to include synthetic ground points
            include_synthetic = features_cfg.get(
                "include_synthetic_ground",
                processor_cfg.get("include_synthetic_ground", True),
            )

            # Compute is_ground feature with statistics
            is_ground, stats = compute_is_ground_with_stats(
                classification=classification,
                synthetic_flags=synthetic_flags,
                ground_class=2,  # ASPRS ground class
                include_synthetic=include_synthetic,
                verbose=False,  # Don't log here, we'll log summary below
            )

            # Add to features
            all_features["is_ground"] = is_ground

            # Log summary
            if stats["synthetic_ground"] > 0:
                logger.info(
                    f"  ✓ is_ground feature: {stats['total_ground']:,} ground points "
                    f"({stats['ground_percentage']:.1f}%) | "
                    f"{stats['synthetic_ground']:,} from DTM "
                    f"({stats['synthetic_percentage']:.1f}%)"
                )
            else:
                logger.info(
                    f"  ✓ is_ground feature: {stats['total_ground']:,} ground points "
                    f"({stats['ground_percentage']:.1f}%)"
                )

        except ImportError as e:
            logger.warning("  ⚠️  is_ground module not available")
            logger.debug(f"  Import error: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"  ⚠️  Invalid data for is_ground computation: {e}")
        except Exception as e:
            error = FeatureComputationError.create(
                feature_name="is_ground",
                error=e,
                num_points=len(tile_data.get("classification", []))
            )
            logger.warning(str(error))

    def _add_architectural_style(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ):
        """
        Add architectural style features if requested.

        Args:
            tile_data: Tile data dict
            all_features: Features dict to update
        """
        processor_cfg = self.config.get("processor", {})
        include_style = processor_cfg.get("include_architectural_style", False)

        if not include_style:
            return

        try:
            from .architectural_styles import compute_architectural_style_features

            points = tile_data["points"]
            classification = tile_data["classification"]
            encoding = processor_cfg.get("style_encoding", "constant")

            style_features = compute_architectural_style_features(
                points=points, classification=classification, encoding=encoding
            )

            all_features["architectural_style"] = style_features
            logger.info(f"  ✓ Added architectural style features (encoding={encoding})")

        except ImportError as e:
            logger.warning("  ⚠️  Architectural style module not available")
            logger.debug(f"  Import error: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"  ❌ Invalid configuration for architectural style: {e}")
        except Exception as e:
            error = FeatureComputationError.create(
                feature_name="architectural_style",
                error=e,
                num_points=len(tile_data.get("points", []))
            )
            logger.error(str(error))

    def _apply_building_buffers(
        self, buildings_gdf: "gpd.GeoDataFrame", points: np.ndarray
    ) -> "gpd.GeoDataFrame":
        """
        Apply adaptive buffers to building geometries for better facade/wall capture.

        This function implements the adaptive buffering strategy specified in the config:
        - Base buffer_distance
        - Adaptive buffer range (min/max) based on building characteristics
        - Vertical buffer for height tolerance
        - Horizontal buffers for ground and upper levels

        Args:
            buildings_gdf: GeoDataFrame with building polygons
            points: Point cloud for adaptive buffer computation

        Returns:
            GeoDataFrame with buffered building polygons

        Config Parameters Used:
            ground_truth.bd_topo.features.buildings.buffer_distance: Base buffer (m)
            ground_truth.bd_topo.features.buildings.enable_adaptive_buffer: Enable adaptive buffering
            ground_truth.bd_topo.features.buildings.adaptive_buffer_min: Minimum adaptive buffer (m)
            ground_truth.bd_topo.features.buildings.adaptive_buffer_max: Maximum adaptive buffer (m)
        """
        if buildings_gdf is None or len(buildings_gdf) == 0:
            return buildings_gdf

        # Get building configuration from ground_truth section (not data_sources)
        ground_truth_cfg = self.config.get("ground_truth", {})
        bd_topo_cfg = ground_truth_cfg.get("bd_topo", {})
        features_cfg = bd_topo_cfg.get("features", {})
        buildings_cfg = features_cfg.get("buildings", {})

        # Base buffer distance
        base_buffer = buildings_cfg.get("buffer_distance", 1.0)

        # Adaptive buffer settings
        enable_adaptive = buildings_cfg.get("enable_adaptive_buffer", False)
        adaptive_min = buildings_cfg.get("adaptive_buffer_min", 1.0)
        adaptive_max = buildings_cfg.get("adaptive_buffer_max", 8.0)

        logger.info(
            f"  🏢 Applying building buffers: base={base_buffer}m, "
            f"adaptive={enable_adaptive} ({adaptive_min}-{adaptive_max}m)"
        )

        # Create buffered copy
        buffered_gdf = buildings_gdf.copy()

        if enable_adaptive and len(points) > 0:
            # Adaptive buffering based on point density and building characteristics
            # ⚡ OPTIMIZED: Vectorized computation instead of .iterrows() loop (5-10x faster)
            try:
                import geopandas as gpd
                from shapely.geometry import Point as ShapelyPoint
                from shapely.strtree import STRtree

                # Vectorized computation of building metrics
                perimeters = buildings_gdf.geometry.length.values
                areas = buildings_gdf.geometry.area.values

                # Compute adaptive buffer based on building size (vectorized)
                # Larger/more complex buildings get larger buffers for facades
                size_factors = np.clip(perimeters / 100.0, 0.0, 1.0)  # 0-1 based on perimeter
                
                # Shape complexity: circle has complexity=1, other shapes >1
                complexity_factors = np.clip(
                    (perimeters**2) / (4 * np.pi * areas + 1e-10) - 1.0, 0.0, 1.0
                )

                # Interpolate buffer distances (vectorized)
                adaptive_buffers = adaptive_min + (adaptive_max - adaptive_min) * (
                    0.5 * size_factors + 0.5 * complexity_factors
                )

                # Use max of base buffer and adaptive buffer (vectorized)
                buffer_distances = np.maximum(base_buffer, adaptive_buffers)

                # Apply buffers
                buffered_gdf["geometry"] = buildings_gdf.geometry.buffer(
                    buffer_distances, cap_style=2  # Flat caps for building corners
                )

                avg_buffer = np.mean(buffer_distances)
                logger.info(
                    f"    ✓ Applied adaptive buffers: "
                    f"avg={avg_buffer:.2f}m, range=[{min(buffer_distances):.2f}, {max(buffer_distances):.2f}]m"
                )

            except ImportError as e:
                logger.warning(
                    f"    ⚠️  Adaptive buffering requires geopandas/shapely: {e}"
                )
                # Fallback to base buffer
                buffered_gdf["geometry"] = buildings_gdf.geometry.buffer(
                    base_buffer, cap_style=2
                )
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"    ⚠️  Invalid building geometry data: {e}"
                )
                # Fallback to base buffer
                buffered_gdf["geometry"] = buildings_gdf.geometry.buffer(
                    base_buffer, cap_style=2
                )
            except Exception as e:
                logger.warning(
                    f"    ⚠️  Adaptive buffering failed, using base buffer: {e}"
                )
                logger.debug(f"  Error type: {type(e).__name__}")
                # Fallback to base buffer
                buffered_gdf["geometry"] = buildings_gdf.geometry.buffer(
                    base_buffer, cap_style=2
                )
        else:
            # Simple base buffer
            buffered_gdf["geometry"] = buildings_gdf.geometry.buffer(
                base_buffer, cap_style=2
            )
            logger.info(f"    ✓ Applied uniform buffer: {base_buffer}m")

        return buffered_gdf

    def _add_cluster_id_features(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ):
        """
        Add optional cluster ID features for spatial object grouping.

        Computes two types of cluster IDs (strict polygon containment):
        1. building_cluster_id: Points inside building polygons (NO buffering)
        2. parcel_cluster_id: Points inside cadastral parcel polygons (NO buffering)

        These features are optional and only computed if:
        - compute_building_cluster_id or compute_parcel_cluster_id is True in config
        - Ground truth geometries are available

        IMPORTANT: Uses STRICT polygon containment without buffering.
        Only points strictly inside polygons get cluster IDs.

        Args:
            tile_data: Tile data dict (must contain 'points')
            all_features: Features dict to update
        """
        features_cfg = self.config.get("features", {})

        # Check which cluster ID features are requested
        compute_building_clusters = features_cfg.get(
            "compute_building_cluster_id", False
        )
        compute_parcel_clusters = features_cfg.get("compute_parcel_cluster_id", False)

        if not (compute_building_clusters or compute_parcel_clusters):
            return  # No cluster IDs requested

        try:
            from .compute.cluster_id import (
                compute_building_cluster_ids,
                compute_parcel_cluster_ids,
            )

            points = tile_data["points"]
            ground_truth_features = tile_data.get("ground_truth_features", {})

            # Building cluster IDs (strict containment, no buffering)
            if compute_building_clusters:
                buildings = ground_truth_features.get("buildings")
                if buildings is not None and len(buildings) > 0:
                    logger.info(
                        f"  🏢 Computing building cluster IDs for {len(buildings)} buildings "
                        f"(strict polygon containment, no buffering)..."
                    )

                    # Use original buildings WITHOUT buffering
                    building_ids = compute_building_cluster_ids(
                        points, buildings  # Use unbuffered polygons
                    )
                    all_features["building_cluster_id"] = building_ids

                    n_buildings = np.max(building_ids)
                    n_assigned = np.sum(building_ids > 0)
                    pct_assigned = (
                        (n_assigned / len(points) * 100) if len(points) > 0 else 0
                    )
                    logger.info(
                        f"  ✓ Building cluster IDs: {n_buildings} buildings, "
                        f"{n_assigned:,} points assigned ({pct_assigned:.1f}%)"
                    )
                else:
                    logger.warning(
                        "  ⚠️  Building cluster IDs requested but no building geometries available"
                    )

            # Parcel cluster IDs (strict containment, no buffering)
            if compute_parcel_clusters:
                parcels = ground_truth_features.get("parcels")
                if parcels is not None and len(parcels) > 0:
                    logger.info(
                        f"  📦 Computing parcel cluster IDs for {len(parcels)} parcels "
                        f"(strict polygon containment)..."
                    )

                    parcel_ids = compute_parcel_cluster_ids(points, parcels)
                    all_features["parcel_cluster_id"] = parcel_ids

                    n_parcels = np.max(parcel_ids)
                    n_assigned = np.sum(parcel_ids > 0)
                    avg_pts_per_parcel = n_assigned / n_parcels if n_parcels > 0 else 0
                    logger.info(
                        f"  ✓ Parcel cluster IDs: {n_parcels} parcels, {n_assigned:,} points ({avg_pts_per_parcel:.0f} pts/parcel)"
                    )
                else:
                    logger.warning(
                        "  ⚠️  Parcel cluster IDs requested but no parcel geometries available"
                    )

        except ImportError:
            logger.warning(
                "  ⚠️  Cluster ID module not available (requires shapely/geopandas)"
            )
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to compute cluster IDs: {e}")
            logger.debug("  Exception details:", exc_info=True)

    def _add_plane_features(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ):
        """
        Add plane-based features for architectural element detection.

        Computes 8 plane-based features:
        1. plane_id: ID of nearest plane (-1 if not assigned)
        2. plane_type: Type (horizontal=0, vertical=1, inclined=2, none=-1)
        3. distance_to_plane: Distance to plane surface (meters)
        4. plane_area: Area of containing plane (m²)
        5. plane_orientation: Angle from horizontal (degrees)
        6. plane_planarity: Planarity score [0,1]
        7. position_on_plane_u: U coordinate on plane [0,1]
        8. position_on_plane_v: V coordinate on plane [0,1]

        These features enable better classification of:
        - Building facades vs walls
        - Roof types and components
        - Architectural details (dormers, balconies, etc.)

        Only computed if:
        - compute_plane_features is True in config
        - Basic geometric features already computed (normals, planarity)

        Args:
            tile_data: Tile data dict (must contain 'points')
            all_features: Features dict to update
        """
        processor_cfg = self.config.get("processor", {})
        features_cfg = self.config.get("features", {})

        # Check if plane features are requested
        compute_planes = features_cfg.get("compute_plane_features", False)

        # Also check if PLANES mode is active or LOD3_FULL mode (which includes planes)
        if not compute_planes:
            if self.feature_mode == FeatureMode.PLANES:
                compute_planes = True
            elif self.feature_mode == FeatureMode.LOD3_FULL:
                # LOD3_FULL mode includes plane features by default
                compute_planes = features_cfg.get("compute_plane_features", True)

        if not compute_planes:
            return  # No plane features requested

        # Check if required features are available
        if "normals" not in all_features or "planarity" not in all_features:
            logger.warning(
                "  ⚠️  Plane features require normals and planarity - skipping"
            )
            return

        try:
            from ..core.classification.plane_detection import (
                PlaneDetector,
                PlaneFeatureExtractor,
            )

            points = tile_data["points"]
            normals = all_features["normals"]
            planarity = all_features.get("planarity")

            # Use planarity from geometric features, or verticality as fallback
            if planarity is None:
                planarity = all_features.get("verticality")

            if planarity is None:
                logger.warning(
                    "  ⚠️  Cannot compute plane features without planarity/verticality"
                )
                return

            height = all_features.get("height")

            # Get plane detection parameters from config
            plane_config = features_cfg.get("plane_detection", {})
            horizontal_angle_max = plane_config.get("horizontal_angle_max", 10.0)
            vertical_angle_min = plane_config.get("vertical_angle_min", 75.0)
            min_points_per_plane = plane_config.get("min_points_per_plane", 50)
            horizontal_planarity_min = plane_config.get(
                "horizontal_planarity_min", 0.75
            )
            vertical_planarity_min = plane_config.get("vertical_planarity_min", 0.65)
            max_assignment_distance = plane_config.get("max_assignment_distance", 0.5)

            # Initialize plane detector
            detector = PlaneDetector(
                horizontal_angle_max=horizontal_angle_max,
                vertical_angle_min=vertical_angle_min,
                min_points_per_plane=min_points_per_plane,
                horizontal_planarity_min=horizontal_planarity_min,
                vertical_planarity_min=vertical_planarity_min,
            )

            # Initialize plane feature extractor
            extractor = PlaneFeatureExtractor(detector)

            # Extract plane features
            plane_features = extractor.detect_and_assign_planes(
                points=points,
                normals=normals,
                planarity=planarity,
                height=height,
                max_assignment_distance=max_assignment_distance,
            )

            # Add plane features to all_features
            for feature_name, feature_array in plane_features.items():
                all_features[feature_name] = feature_array

            # Log statistics
            n_planes = len(extractor.planes)
            n_assigned = np.sum(plane_features["plane_id"] >= 0)
            pct_assigned = (n_assigned / len(points) * 100) if len(points) > 0 else 0

            logger.info(
                f"  ✓ Plane features: {n_planes} planes detected, "
                f"{n_assigned:,}/{len(points):,} points assigned ({pct_assigned:.1f}%)"
            )

            # Get plane type statistics
            stats = extractor.get_plane_statistics()
            if stats["n_planes"] > 0:
                logger.info(
                    f"     Horizontal: {stats['n_horizontal']}, "
                    f"Vertical: {stats['n_vertical']}, "
                    f"Inclined: {stats['n_inclined']}"
                )

        except ImportError as e:
            logger.warning("  ⚠️  Plane detection module not available")
            logger.debug(f"  Import error: {e}")
        except (ValueError, IndexError, KeyError) as e:
            logger.warning(f"  ⚠️  Invalid data for plane features: {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
        except Exception as e:
            error = FeatureComputationError.create(
                feature_name="plane features",
                error=e,
                num_points=len(tile_data.get("points", []))
            )
            logger.warning(str(error))
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")

    def _add_building_plane_features(
        self, tile_data: Dict[str, Any], all_features: Dict[str, np.ndarray]
    ):
        """
        Add building-plane hierarchical clustering features.

        Computes 7 building-aware plane features:
        1. building_id: ID of containing building
        2. plane_id_local: Local plane ID within building
        3. facade_id: Facade ID for vertical planes (0-3 for N/E/S/W, -1 for non-facade)
        4. distance_to_building_center: Distance from building centroid (meters)
        5. relative_height_in_building: Normalized height within building [0, 1]
        6. n_planes_in_building: Number of planes in building
        7. plane_area_ratio: Plane area / total building surface area

        These features enable:
        - Building-aware ML training
        - Facade-level classification (windows, doors, balconies)
        - LOD3 reconstruction with architectural context
        - Multi-building scene understanding

        Only computed if:
        - compute_building_plane_features is True in config
        - Plane features already computed (plane_id present)
        - Ground truth building footprints available

        Args:
            tile_data: Tile data dict (must contain 'points' and 'ground_truth_features')
            all_features: Features dict to update
        """
        processor_cfg = self.config.get("processor", {})
        features_cfg = self.config.get("features", {})

        # Check if building-plane features are requested
        compute_building_planes = features_cfg.get(
            "compute_building_plane_features", False
        )

        # Also enable for LOD3_FULL mode if plane features are enabled
        if not compute_building_planes:
            if self.feature_mode == FeatureMode.LOD3_FULL:
                compute_building_planes = features_cfg.get(
                    "compute_plane_features", False
                )

        if not compute_building_planes:
            return  # No building-plane features requested

        # Check if plane features are available
        if "plane_id" not in all_features:
            logger.debug(
                "  ⚠️  Building-plane features require plane features - skipping"
            )
            return

        # Check if ground truth buildings are available
        ground_truth_features = tile_data.get("ground_truth_features", {})
        buildings_gdf = ground_truth_features.get("buildings")

        if buildings_gdf is None or len(buildings_gdf) == 0:
            logger.debug(
                "  ⚠️  Building-plane features require building footprints - skipping"
            )
            return

        try:
            from ..core.classification.building.clustering import BuildingPlaneClusterer

            points = tile_data["points"]
            classification = tile_data.get("classification")

            # Get plane features for clustering
            plane_features = {
                "plane_id": all_features["plane_id"],
                "plane_type": all_features.get("plane_type"),
                "plane_area": all_features.get("plane_area"),
                "normals": all_features.get("normals"),
            }

            # Get building clustering parameters
            building_config = features_cfg.get("building_plane_clustering", {})
            min_points_per_plane = building_config.get("min_points_per_plane", 30)
            compute_facade_ids = building_config.get("compute_facade_ids", True)

            # Building classes for filtering (ASPRS codes)
            building_classes = [6]  # Building class

            # Initialize clusterer
            clusterer = BuildingPlaneClusterer(
                min_points_per_plane=min_points_per_plane,
                compute_facade_ids=compute_facade_ids,
            )

            # Cluster points by building and plane
            building_plane_features, clusters = (
                clusterer.cluster_points_by_building_planes(
                    points=points,
                    plane_features=plane_features,
                    buildings_gdf=buildings_gdf,
                    labels=classification,
                    building_classes=building_classes,
                )
            )

            # Add building-plane features to all_features
            for feature_name, feature_array in building_plane_features.items():
                all_features[feature_name] = feature_array

            # Log statistics
            n_clusters = len(clusters)
            n_buildings = len(
                np.unique(
                    building_plane_features["building_id"][
                        building_plane_features["building_id"] >= 0
                    ]
                )
            )
            n_assigned = np.sum(building_plane_features["building_id"] >= 0)

            if n_buildings > 0:
                avg_planes_per_building = n_clusters / n_buildings
                logger.info(
                    f"  ✓ Building-plane features: {n_buildings} buildings, "
                    f"{n_clusters} plane clusters ({avg_planes_per_building:.1f} planes/building)"
                )
                logger.info(
                    f"     {n_assigned:,}/{len(points):,} points assigned to buildings"
                )
            else:
                logger.info(
                    "  ✓ Building-plane features computed (no buildings with planes)"
                )

        except ImportError as e:
            logger.warning("  ⚠️  Building plane clustering module not available")
            logger.debug(f"  Import error: {e}")
        except (ValueError, IndexError, KeyError) as e:
            logger.warning(f"  ⚠️  Invalid data for building-plane features: {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
        except Exception as e:
            error = FeatureComputationError.create(
                feature_name="building-plane features",
                error=e,
                num_points=len(tile_data.get("points", []))
            )
            logger.warning(str(error))
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")

    # =========================================================================
    # PROPERTIES & UTILITIES
    # =========================================================================

    @property
    def has_rgb(self) -> bool:
        """Check if RGB fetcher is available."""
        return self.rgb_fetcher is not None

    @property
    def has_infrared(self) -> bool:
        """Check if infrared fetcher is available."""
        return self.infrared_fetcher is not None

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return (
            f"FeatureOrchestrator("
            f"strategy={self.strategy_name}, "
            f"mode={self.feature_mode.value}, "
            f"gpu={self.gpu_available}, "
            f"rgb={self.has_rgb}, "
            f"nir={self.has_infrared})"
        )

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, "_rgb_nir_executor"):
                self._rgb_nir_executor.shutdown(wait=False)
            if hasattr(self, "_feature_executor"):
                self._feature_executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors

    def __getstate__(self):
        """
        Custom serialization for multiprocessing compatibility.

        Excludes non-picklable thread pool executors.
        """
        state = self.__dict__.copy()
        # Remove non-picklable thread pool executors
        state["_rgb_nir_executor"] = None
        state["_feature_executor"] = None
        return state

    def __setstate__(self, state):
        """
        Custom deserialization for multiprocessing compatibility.

        Reinitializes thread pools after unpickling.
        """
        self.__dict__.update(state)
        # Shutdown any existing executors (shouldn't exist but be safe)
        if hasattr(self, "_rgb_nir_executor") and self._rgb_nir_executor is not None:
            try:
                self._rgb_nir_executor.shutdown(wait=False)
            except Exception:
                pass
        if hasattr(self, "_feature_executor") and self._feature_executor is not None:
            try:
                self._feature_executor.shutdown(wait=False)
            except Exception:
                pass
        # Reinitialize thread pools
        self._init_parallel_processing()
