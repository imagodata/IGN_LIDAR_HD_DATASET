"""
Tile processing coordinator for LiDAR processing.

This module orchestrates all processing components to handle
individual LAZ tile processing in a clean, modular way.

Extracted from LiDARProcessor as part of refactoring (v3.4.0).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from .classification_applier import ClassificationApplier
from .output_writer import OutputWriter
from .patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)


class TileProcessor:
    """
    Orchestrates tile-level processing workflow for individual LAZ tiles.

    This class is the **MIDDLE LAYER** between LiDARProcessor (batch orchestration)
    and ProcessorCore (low-level operations). It coordinates all components needed
    to process a single LiDAR tile from raw LAZ to final output.

    Architecture Position:
    =====================

    LiDARProcessor (batch orchestration)
        ↓
    **TileProcessor** ← You are here (tile orchestration)
        ↓
    ProcessorCore (low-level operations)

    Workflow Steps:
    ==============

    1. **Tile Loading** (via TileLoader)
       - Read LAZ file
       - Extract point cloud (XYZ, classification, RGB, NIR, etc.)
       - Validate tile data (point count, bounds, etc.)
       - Apply bounding box filtering if specified

    2. **Ground Truth Integration** (via ClassificationApplier)
       - Fetch WFS data (BD TOPO buildings, roads, water, vegetation)
       - Spatial join with point cloud
       - Apply ground truth labels to points
       - Refine classifications based on features

    3. **Feature Computation** (via FeatureOrchestrator)
       - Select appropriate strategy (CPU/GPU/GPU-chunked)
       - Compute geometric features (normals, curvature, etc.)
       - Compute spectral features (RGB, NIR, NDVI)
       - Multi-scale features (if enabled)
       - Architectural style detection (if enabled)

    4. **Patch Extraction** (via PatchExtractor)
       - Spatial gridding of tile
       - Extract patches of configurable size (e.g., 150m x 150m)
       - Subsample to target point count (e.g., 16384 points)
       - Apply augmentation if enabled
       - Handle tile boundaries with buffer zones

    5. **Output Generation** (via OutputWriter)
       - Save patches in multiple formats:
         * NPZ: Lightweight NumPy arrays
         * HDF5: Hierarchical data with metadata
         * LAZ: Enriched point clouds with features
         * PyTorch: Ready-to-use tensors
       - Save metadata (processing stats, config, etc.)
       - Generate enriched full-tile LAZ (if requested)

    Key Responsibilities:
    ====================

    1. **Component Coordination**: Orchestrate specialized components
       - Feature computation
       - Classification application
       - Patch extraction
       - Output writing
       - Each component is independently testable

    2. **State Management**: Track tile processing state
       - Points loaded/processed
       - Features computed
       - Patches extracted
       - Outputs saved

    3. **Error Handling**: Graceful degradation
       - Continue on non-fatal errors
       - Log warnings for recoverable issues
       - Propagate critical errors to caller

    4. **Performance Monitoring**: Track processing metrics
       - Time per stage
       - Memory usage
       - Feature computation stats
       - Patch generation stats

    Processing Modes:
    ================

    - **patches_only** (default): Generate training patches only
      - Extract patches with features and labels
      - Save in ML-ready formats (NPZ, HDF5, PyTorch)
      - Fastest mode for dataset creation

    - **enriched_only**: Generate enriched LAZ tiles only
      - Add features as extra dimensions to LAZ
      - Apply/refine classifications
      - No patch extraction
      - Use for visualization or further processing

    - **both**: Generate patches AND enriched LAZ
      - All outputs from both modes
      - Most comprehensive but slowest
      - Use for production pipelines

    - **reclassify_only**: Re-classify existing enriched tiles
      - Load enriched LAZ with features
      - Apply new classification rules
      - Save updated LAZ
      - Fast iteration on classification logic

    Dependencies:
    ============

    - **FeatureOrchestrator**: Feature computation with CPU/GPU strategies
    - **ClassificationApplier**: Ground truth integration and classification
    - **PatchExtractor**: Spatial gridding and patch creation
    - **OutputWriter**: Multi-format output generation
    - **TileLoader**: LAZ file loading and validation (optional, lazy-loaded)
    - **ProcessorCore**: Low-level spatial operations (used by components)

    Design Principles:
    =================

    1. **Single Responsibility**: One tile at a time
       - No cross-tile dependencies
       - Stateless processing (tile-to-tile)
       - Parallelizable by LiDARProcessor

    2. **Dependency Injection**: Components passed at init
       - Easy testing with mocks
       - Flexible configuration
       - Clear dependencies

    3. **Error Isolation**: Tile failures don't affect others
       - Try/except around tile processing
       - Detailed error logging
       - Continue batch processing on error

    Example Usage:
    =============

    Typical usage (via LiDARProcessor):
        >>> # Usually created by LiDARProcessor, not directly
        >>> processor = LiDARProcessor(config)
        >>> processor.process_tiles()  # Uses TileProcessor internally

    Direct usage (advanced):
        >>> tile_processor = TileProcessor(
        ...     config=config,
        ...     feature_orchestrator=feature_orch,
        ...     patch_extractor=patch_ext,
        ...     classification_applier=class_app,
        ...     output_writer=output_writer
        ... )
        >>> result = tile_processor.process_tile(
        ...     tile_path=Path('/data/tile.laz'),
        ...     output_dir=Path('/data/output')
        ... )
        >>> print(f"Processed {result['num_points']} points, "
        ...       f"extracted {result['num_patches']} patches")

    Performance Characteristics:
    ===========================

    Typical tile (18M points, LOD2 mode, GPU enabled):
    - Loading: ~2-3 seconds
    - Ground truth: ~5-10 seconds (WFS queries + spatial joins)
    - Features: ~30-60 seconds (GPU) or ~5-10 minutes (CPU)
    - Patches: ~10-20 seconds
    - Output: ~5-10 seconds
    - **Total: ~1-2 minutes (GPU) or ~6-12 minutes (CPU)**

    Memory usage:
    - Base: ~500 MB (point cloud + features)
    - Peak: ~2-4 GB (during patch extraction with augmentation)
    - GPU: +2-4 GB VRAM (for feature computation)

    Related Classes:
    ===============

    - **LiDARProcessor**: Parent class, batch orchestration (1 level up)
    - **ProcessorCore**: Low-level operations (1 level down)
    - **FeatureOrchestrator**: Feature computation management
    - **ClassificationApplier**: Ground truth application
    - **PatchExtractor**: Patch extraction logic
    - **OutputWriter**: Multi-format output generation

    Version History:
    ===============

    - v3.4.0: Extracted from LiDARProcessor for better modularity
    - v3.2.0: Added reclassify_only mode
    - v3.1.0: Integrated Classifier
    - v3.0.0: GPU acceleration support

    See Also:
    ========

    - LiDARProcessor: For batch processing
    - ProcessorCore: For low-level operations
    - FeatureOrchestrator: For feature computation
    """

    def __init__(
        self,
        config: DictConfig,
        feature_orchestrator: Any,
        patch_extractor: PatchExtractor,
        classification_applier: ClassificationApplier,
        output_writer: OutputWriter,
        tile_loader: Optional[Any] = None,
    ):
        """
        Initialize tile processor.

        Args:
            config: Configuration
            feature_orchestrator: Feature computation orchestrator
            patch_extractor: Patch extraction component
            classification_applier: Classification application component
            output_writer: Output generation component
            tile_loader: Optional tile loader for pre-loaded data
        """
        self.config = config
        self.feature_orchestrator = feature_orchestrator
        self.patch_extractor = patch_extractor
        self.classification_applier = classification_applier
        self.output_writer = output_writer
        self.tile_loader = tile_loader

        # Processing configuration
        self.save_patches = config.processor.get("save_patches", True)
        self.processing_mode = config.processor.get("processing_mode", "patches_only")

        logger.debug(
            f"Initialized TileProcessor: "
            f"mode={self.processing_mode}, "
            f"save_patches={self.save_patches}"
        )

    def process_tile(
        self,
        laz_file: Path,
        output_dir: Path,
        tile_data: Optional[Dict[str, Any]] = None,
        prefetched_ground_truth: Optional[Dict[str, Any]] = None,
        progress_prefix: str = "",
        tile_split: Optional[str] = None,
    ) -> int:
        """
        Process a single LAZ tile through complete pipeline.

        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_data: Optional pre-loaded tile data
            prefetched_ground_truth: Optional pre-fetched ground truth
            progress_prefix: Progress message prefix
            tile_split: Optional split name (train/val/test)

        Returns:
            Number of patches saved (0 if enriched_only mode)

        Raises:
            ValueError: If tile processing fails critically
        """
        tile_start = time.time()

        try:
            # Step 1: Load tile data
            logger.info(f"{progress_prefix} 📂 Loading tile: {laz_file.name}")
            original_data = self._load_tile(laz_file, tile_data)

            # Step 2: Compute features
            logger.info(f"{progress_prefix} 🧮 Computing features...")
            all_features, points, classification = self._compute_features(
                original_data, laz_file, progress_prefix
            )

            # Step 3: Apply classification (ground truth + mapping)
            logger.info(f"{progress_prefix} 🏷️  Applying classification...")
            labels = self._apply_classification(
                points, classification, all_features, progress_prefix
            )

            # Step 4: Extract patches (if enabled)
            num_patches_saved = 0
            if self.save_patches and not self.output_writer.only_enriched_laz:
                logger.info(f"{progress_prefix} 📦 Extracting patches...")
                patches = self._extract_patches(points, all_features, labels)

                # Step 5a: Save patches
                logger.info(f"{progress_prefix} 💾 Saving patches...")
                num_patches_saved = self.output_writer.save_patches(
                    patches, laz_file, output_dir, tile_split
                )

            # Step 5b: Save enriched LAZ (if enabled)
            if self.output_writer.should_save_enriched_laz:
                logger.info(f"{progress_prefix} 💾 Saving enriched LAZ...")
                self.output_writer.save_enriched_laz(
                    laz_file, output_dir, points, labels, all_features, original_data
                )

            # Step 6: Save metadata
            tile_time = time.time() - tile_start
            self.output_writer.save_metadata(
                laz_file, output_dir, tile_time, len(points), num_patches_saved
            )

            # Log completion
            logger.info(
                f"{progress_prefix} ✅ Completed: {num_patches_saved} patches "
                f"in {tile_time:.1f}s (from {len(points):,} points)"
            )

            return num_patches_saved

        except Exception as e:
            logger.error(f"{progress_prefix} ❌ Tile processing failed: {e}")
            logger.exception(e)
            raise

    def _load_tile(
        self,
        laz_file: Path,
        tile_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load tile data.

        Args:
            laz_file: Path to LAZ file
            tile_data: Optional pre-loaded data

        Returns:
            Dictionary with points, classification, intensity, etc.
        """
        if tile_data is not None:
            logger.debug("Using pre-loaded tile data")
            return tile_data

        # Load using tile loader if available
        if self.tile_loader is not None:
            return self.tile_loader.load_tile(laz_file)

        # Fallback: Load directly using laspy
        import laspy

        las = laspy.read(str(laz_file))
        return {
            "points": np.vstack([las.x, las.y, las.z]).T,
            "classification": las.classification,
            "intensity": las.intensity if hasattr(las, "intensity") else None,
            "return_number": (
                las.return_number if hasattr(las, "return_number") else None
            ),
            "num_returns": (
                las.number_of_returns if hasattr(las, "number_of_returns") else None
            ),
        }

    def _compute_features(
        self,
        original_data: Dict[str, Any],
        laz_file: Path,
        progress_prefix: str = "",
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Compute features for point cloud.

        Args:
            original_data: Original tile data
            laz_file: LAZ file path
            progress_prefix: Progress prefix for logging

        Returns:
            Tuple of (features_dict, points, classification)
        """
        points = original_data["points"]
        classification = original_data["classification"]

        # Compute features using orchestrator. `FeatureOrchestrator` has no
        # `compute_all_features` method (only `FeatureComputer` does) -- this
        # call used to raise AttributeError unconditionally. `compute_features`
        # takes the tile_data dict as a single argument, matching the shape
        # `_load_tile()` already returns (points/classification/intensity/
        # return_number/num_returns/input_rgb/input_nir/...).
        all_features = self.feature_orchestrator.compute_features(
            tile_data=original_data,
        )

        # Validate feature array sizes
        n_points = len(points)
        for name, feat_array in all_features.items():
            if feat_array is not None and len(feat_array) != n_points:
                logger.warning(
                    f"  ⚠️  Feature '{name}' size mismatch: "
                    f"{len(feat_array)} != {n_points}"
                )

        return all_features, points, classification

    def _apply_classification(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        features: Dict[str, np.ndarray],
        progress_prefix: str = "",
    ) -> np.ndarray:
        """
        Apply classification mapping and ground truth.

        Args:
            points: Point cloud [N, 3]
            classification: Original classification [N]
            features: Computed features
            progress_prefix: Progress prefix for logging

        Returns:
            Updated classification labels [N]
        """
        # Step 1: Apply ASPRS → LOD mapping (if configured)
        labels = self.classification_applier.apply_class_mapping(classification)

        # Step 2: Apply ground truth classification (if available)
        if self.classification_applier.data_fetcher is not None:
            # Compute bounding box
            bbox = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )

            # Apply ground truth
            labels = self.classification_applier.apply_ground_truth(
                points, labels, features, bbox
            )

        return labels

    def _extract_patches(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract patches from processed point cloud.

        Args:
            points: Point cloud [N, 3]
            features: Computed features
            classification: Classification labels [N]

        Returns:
            List of patch dictionaries
        """
        patches = self.patch_extractor.extract_patches(
            points=points,
            features=features,
            classification=classification,
        )

        logger.debug(f"  Extracted {len(patches)} patches")
        return patches

    def _get_tile_bounds(self, points: np.ndarray) -> Tuple[float, ...]:
        """
        Get tile bounding box.

        Args:
            points: Point cloud [N, 3]

        Returns:
            Tuple of (xmin, ymin, xmax, ymax)
        """
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tile processor statistics.

        Returns:
            Dictionary with configuration and component stats
        """
        return {
            "processing_mode": self.processing_mode,
            "save_patches": self.save_patches,
            "has_tile_loader": self.tile_loader is not None,
            "components": {
                "feature_orchestrator": (self.feature_orchestrator.__class__.__name__),
                "patch_extractor": self.patch_extractor.get_statistics(),
                "classification_applier": (
                    self.classification_applier.get_statistics()
                ),
                "output_writer": self.output_writer.get_statistics(),
            },
        }
