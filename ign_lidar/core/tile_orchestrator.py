"""
Tile Processing Orchestration

This module handles the core tile processing logic, coordinating data loading,
feature computation, classification, patch extraction, and output generation.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..classification_schema import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from ..features.architectural_styles import get_architectural_style_id
from ..features.orchestrator import FeatureOrchestrator
from ..io.metadata import MetadataManager
from .classification import Classifier, refine_classification
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
from .classification.reclassifier import Reclassifier
from .skip_checker import PatchSkipChecker

logger = logging.getLogger(__name__)


class TileOrchestrator:
    """
    Orchestrates tile-level processing operations.
    
    Responsibilities:
    - Coordinate tile data preparation (loading, augmentation, preprocessing)
    - Manage feature computation workflow
    - Apply classification and refinement
    - Extract patches with augmentation
    - Generate outputs in multiple formats
    
    This class extracts the complex tile processing logic from LiDARProcessor,
    providing a cleaner separation of concerns and better maintainability.
    """

    def __init__(
        self,
        config: DictConfig,
        feature_orchestrator: FeatureOrchestrator,
        classifier: Optional[Classifier] = None,
        reclassifier: Optional[Reclassifier] = None,
        lod_level: str = "LOD2",
        class_mapping: Optional[Dict] = None,
        default_class: int = 14,
        data_fetcher: Optional[Any] = None,
    ):
        """
        Initialize TileOrchestrator.
        
        Args:
            config: Hydra configuration
            feature_orchestrator: FeatureOrchestrator instance for feature computation
            classifier: Optional Classifier for ground truth classification
            reclassifier: Optional Reclassifier for refined classification
            lod_level: Level of detail ("LOD2", "LOD3", or "ASPRS")
            class_mapping: Optional class mapping dictionary
            default_class: Default class code for unmapped classes
            data_fetcher: Optional DataFetcher for ground truth data (buildings, etc.)
        """
        self.config = config
        # Lazy OutputWriter: the single save path that handles every configured
        # format, including 'ptv3_pointcept' (coord/feat/segment.npy + meta.json).
        self._output_writer = None
        self.feature_orchestrator = feature_orchestrator
        self.classifier = classifier
        self.data_fetcher = data_fetcher
        self.reclassifier = reclassifier
        self.lod_level = lod_level
        self.class_mapping = class_mapping
        self.default_class = default_class
        
        # Extract commonly used config values
        self.processing_mode = config.processor.processing_mode
        # Fallback: processor.output_format -> output.format -> 'npz'
        self.output_format = OmegaConf.select(
            config, "processor.output_format",
            default=OmegaConf.select(config, "output.format", default="npz")
        )
        self.include_architectural_style = OmegaConf.select(
            config, "processor.include_architectural_style", default=False
        )
        self.augment = OmegaConf.select(config, "processor.augment", default=False)
        self.num_augmentations = OmegaConf.select(config, "processor.num_augmentations", default=0)
        
        # Initialize skip checker
        processing_mode = config.processor.processing_mode
        save_enriched = OmegaConf.select(config, "output.enriched_laz.enabled", default=False)
        only_enriched = processing_mode == "enriched_only"
        
        self.skip_checker = PatchSkipChecker(
            output_format=self.output_format,
            architecture=config.processor.architecture,
            num_augmentations=self.num_augmentations,
            augment=self.augment,
            validate_content=True,
            only_enriched_laz=only_enriched,
        )

    def process_tile_core(
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

        ⚡ OPTIMIZATION: This method accepts pre-loaded tile_data to avoid redundant I/O.

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
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""
        tile_start = time.time()

        # 1. Load architectural style metadata if requested
        architectural_style_id, multi_styles, tile_metadata = (
            self._load_architectural_metadata(laz_file)
        )

        # 2. Extract and prepare tile data
        points, intensity, return_number, num_returns, classification, input_rgb, input_nir, input_ndvi, enriched_features = (
            self._extract_tile_data(tile_data)
        )

        # Store original data for all versions
        original_data = self._create_original_data_dict(
            points, intensity, return_number, num_returns, classification,
            input_rgb, input_nir, input_ndvi, enriched_features, tile_data
        )

        # 3. Augment ground points with DTM if enabled (BEFORE feature computation)
        points, classification, intensity, return_number, num_returns, input_rgb, input_nir, input_ndvi = (
            self._augment_ground_with_dtm_if_enabled(
                points, classification, intensity, return_number, num_returns,
                input_rgb, input_nir, input_ndvi
            )
        )

        # 4. Compute features
        logger.info(f"{progress_prefix} Computing features...")
        features_start = time.time()
        
        features = self.feature_orchestrator.compute_features(
            tile_data={
                "points": points,
                "intensity": intensity,
                "return_number": return_number,
                "num_returns": num_returns,
                "classification": classification,
                "input_rgb": input_rgb,
                "input_nir": input_nir,
                "input_ndvi": input_ndvi,
                "enriched_features": enriched_features,
            }
        )
        
        features_time = time.time() - features_start
        logger.info(f"{progress_prefix} ✓ Features computed in {features_time:.1f}s")

        # 5. Apply classification and refinement
        # Extract ground truth from tile_data if available
        ground_truth = tile_data.get("ground_truth")
        classification = self._apply_classification_and_refinement(
            points, features, classification, ground_truth, progress_prefix
        )

        # 6. Extract patches
        num_patches = self._extract_and_save_patches(
            laz_file=laz_file,
            output_dir=output_dir,
            points=points,
            features=features,
            classification=classification,
            intensity=intensity,
            return_number=return_number,
            original_data=original_data,
            architectural_style_id=architectural_style_id,
            multi_styles=multi_styles,
            tile_metadata=tile_metadata,
            skip_existing=skip_existing,
            progress_prefix=progress_prefix,
        )

        # 7. Log summary
        tile_time = time.time() - tile_start
        logger.info(
            f"{progress_prefix} ✓ Tile processed in {tile_time:.1f}s "
            f"({num_patches} patches created)"
        )

        # 8. Cleanup
        gc.collect()
        
        return num_patches

    def _load_architectural_metadata(
        self, laz_file: Path
    ) -> Tuple[int, Optional[list], Optional[dict]]:
        """
        Load architectural style metadata for the tile.
        
        Args:
            laz_file: Path to LAZ file
            
        Returns:
            Tuple of (architectural_style_id, multi_styles, tile_metadata)
        """
        architectural_style_id = 0  # Default: unknown
        multi_styles = None
        tile_metadata = None

        if self.include_architectural_style:
            metadata_mgr = MetadataManager(laz_file.parent)
            tile_metadata = metadata_mgr.load_tile_metadata(laz_file)

            if tile_metadata:
                # Check for new multi-label styles
                if "architectural_styles" in tile_metadata:
                    multi_styles = tile_metadata["architectural_styles"]
                    style_names = [s.get("style_name", "?") for s in multi_styles]
                    logger.info(f"  🏛️  Multi-style: {', '.join(style_names)}")
                else:
                    # Fall back to single style (legacy)
                    characteristics = tile_metadata.get("characteristics", [])
                    category = tile_metadata.get("location", {}).get("category")
                    architectural_style_id = get_architectural_style_id(
                        characteristics=characteristics, category=category
                    )
                    loc_name = tile_metadata.get("location", {}).get("name", "?")
                    logger.info(f"  🏛️  Style: {architectural_style_id} ({loc_name})")
            else:
                logger.debug(f"  No metadata for {laz_file.name}, style=0")

        return architectural_style_id, multi_styles, tile_metadata

    def _extract_tile_data(self, tile_data: dict) -> Tuple:
        """
        Extract tile data arrays from TileLoader output.
        
        Args:
            tile_data: Dictionary from TileLoader
            
        Returns:
            Tuple of (points, intensity, return_number, num_returns, classification,
                     input_rgb, input_nir, input_ndvi, enriched_features)
        """
        points = tile_data["points"]
        intensity = tile_data["intensity"]
        return_number = tile_data["return_number"]
        # .get(): tile_data dicts built outside TileLoader (tests, FRACTAL
        # conversion) predate this key; a missing num_returns just zero-fills
        # its PTv3 column instead of blowing up the whole tile.
        num_returns = tile_data.get("num_returns")
        classification = tile_data["classification"]
        input_rgb = tile_data.get("input_rgb")
        input_nir = tile_data.get("input_nir")
        input_ndvi = tile_data.get("input_ndvi")
        enriched_features = tile_data.get("enriched_features", {})

        return (
            points, intensity, return_number, num_returns, classification,
            input_rgb, input_nir, input_ndvi, enriched_features
        )

    def _create_original_data_dict(
        self,
        points: np.ndarray,
        intensity: Optional[np.ndarray],
        return_number: Optional[np.ndarray],
        num_returns: Optional[np.ndarray],
        classification: np.ndarray,
        input_rgb: Optional[np.ndarray],
        input_nir: Optional[np.ndarray],
        input_ndvi: Optional[np.ndarray],
        enriched_features: Dict,
        tile_data: dict,
    ) -> Dict:
        """
        Create original data dictionary for backup.
        
        Args:
            points: Point cloud coordinates
            intensity: Intensity values
            return_number: Return number values
            num_returns: Total returns per emitting pulse
            classification: Classification codes
            input_rgb: RGB colors
            input_nir: NIR values
            input_ndvi: NDVI values
            enriched_features: Pre-computed features
            tile_data: Raw tile data from loader

        Returns:
            Dictionary with original data
        """
        return {
            "points": points,
            "intensity": intensity,
            "return_number": return_number,
            "num_returns": num_returns,
            "classification": classification,
            "input_rgb": input_rgb,
            "input_nir": input_nir,
            "input_ndvi": input_ndvi,
            "enriched_features": enriched_features,
            "las": tile_data.get("las"),
            "header": tile_data.get("header"),
        }

    def _augment_ground_with_dtm_if_enabled(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        intensity: Optional[np.ndarray],
        return_number: Optional[np.ndarray],
        num_returns: Optional[np.ndarray],
        input_rgb: Optional[np.ndarray],
        input_nir: Optional[np.ndarray],
        input_ndvi: Optional[np.ndarray],
    ) -> Tuple:
        """
        Augment ground points with DTM if enabled in config.
        
        This must happen BEFORE feature computation so features are computed
        on all points including synthetic ground points.
        
        Args:
            points: Point coordinates [N, 3]
            classification: Classification codes [N]
            intensity: Intensity values [N] or None
            return_number: Return numbers [N] or None
            num_returns: Total returns per pulse [N] or None
            input_rgb: RGB colors [N, 3] or None
            input_nir: NIR values [N] or None
            input_ndvi: NDVI values [N] or None

        Returns:
            Tuple of (points, classification, intensity, return_number,
                     num_returns, input_rgb, input_nir, input_ndvi)
            with augmented data
        """
        rge_alti_enabled = OmegaConf.select(
            self.config, "data_sources.rge_alti.enabled", default=False
        )
        augment_ground = OmegaConf.select(
            self.config, "ground_truth.rge_alti.augment_ground", default=False
        )

        if not (rge_alti_enabled and augment_ground):
            return (
                points, classification, intensity, return_number, num_returns,
                input_rgb, input_nir, input_ndvi
            )

        logger.info("  🌍 Augmenting ground points with RGE ALTI DTM...")
        
        try:
            # Calculate bounding box from points
            bbox = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
            
            # Augment ground points using DTM
            points_augmented, classification_augmented = self._augment_ground_with_dtm(
                points=points,
                classification=classification,
                bbox=bbox
            )
            
            # Update point cloud with augmented data
            n_added = len(points_augmented) - len(points)
            if n_added > 0:
                logger.info(f"  ✅ Added {n_added:,} synthetic ground points from DTM")
                
                # Extend other arrays with null/default values for synthetic points
                if intensity is not None:
                    intensity = np.concatenate([intensity, np.zeros(n_added, dtype=intensity.dtype)])
                if return_number is not None:
                    return_number = np.concatenate([return_number, np.ones(n_added, dtype=return_number.dtype)])
                # A synthetic DTM point stands for a lone ground echo:
                # return_number = num_returns = 1 (not 0, which is not a legal
                # LAS value and would read as "unknown" to the network).
                if num_returns is not None:
                    num_returns = np.concatenate([num_returns, np.ones(n_added, dtype=num_returns.dtype)])
                if input_rgb is not None:
                    input_rgb = np.concatenate([input_rgb, np.zeros((n_added, 3), dtype=input_rgb.dtype)])
                if input_nir is not None:
                    input_nir = np.concatenate([input_nir, np.zeros(n_added, dtype=input_nir.dtype)])
                if input_ndvi is not None:
                    input_ndvi = np.concatenate([input_ndvi, np.zeros(n_added, dtype=input_ndvi.dtype)])
                
                return (
                    points_augmented, classification_augmented, intensity, return_number,
                    num_returns, input_rgb, input_nir, input_ndvi
                )
        except Exception as e:
            logger.warning(f"  ⚠️  DTM augmentation failed: {e}")

        return (
            points, classification, intensity, return_number, num_returns,
            input_rgb, input_nir, input_ndvi
        )

    def _apply_classification_and_refinement(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        ground_truth: Optional[Any] = None,
        progress_prefix: str = "",
    ) -> np.ndarray:
        """
        Apply classification and refinement if classifier is available.
        
        Args:
            points: Point coordinates
            features: Computed features
            classification: Current classification
            ground_truth: Optional ground truth data (GeoDataFrame or dict)
            progress_prefix: Prefix for log messages
            
        Returns:
            Updated classification array
        """
        if self.classifier is None:
            logger.debug(f"{progress_prefix} No classifier configured, keeping original classification")
            return classification

        logger.info(f"{progress_prefix} Applying classification with {len(points)} points...")
        
        try:
            # Use standardized v3.2+ classify() method
            result = self.classifier.classify(
                points=points,
                features=features,
                ground_truth=ground_truth,
                verbose=False  # Reduce logging noise
            )
            
            # Extract labels from result
            labels = result.labels
            
            # Log classification statistics
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"{progress_prefix} Classification complete:")
            for cls, count in zip(unique, counts):
                percentage = 100.0 * count / len(labels)
                logger.info(f"  Class {cls}: {count:,} points ({percentage:.1f}%)")
            
            return labels
            
        except Exception as e:
            logger.error(f"{progress_prefix} Classification failed: {e}", exc_info=True)
            logger.warning(f"{progress_prefix} Falling back to original classification")
            return classification

    def _extract_and_save_patches(
        self,
        laz_file: Path,
        output_dir: Path,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        intensity: Optional[np.ndarray],
        return_number: Optional[np.ndarray],
        original_data: Dict,
        architectural_style_id: int,
        multi_styles: Optional[list],
        tile_metadata: Optional[dict],
        skip_existing: bool,
        progress_prefix: str = "",
    ) -> int:
        """
        Extract patches and save them in the specified format(s).
        
        Args:
            laz_file: Source LAZ file path
            output_dir: Output directory
            points: Point coordinates
            features: Computed features
            classification: Classification codes
            intensity: Intensity values
            return_number: Return numbers
            original_data: Original tile data for backup
            architectural_style_id: Architectural style ID
            multi_styles: Multi-label styles (if any)
            tile_metadata: Tile metadata dictionary
            skip_existing: Whether to skip existing patches
            progress_prefix: Prefix for log messages
            
        Returns:
            Number of patches created
        """
        # Check if should skip
        # NOTE: should_skip_tile() returns a (bool, dict) tuple. It MUST be
        # unpacked — testing the tuple directly is always truthy and would skip
        # every tile (even with zero existing patches → 0 patches ever created).
        if skip_existing:
            should_skip, skip_info = self.skip_checker.should_skip_tile(
                laz_file, output_dir
            )
            if should_skip:
                logger.info(
                    f"{progress_prefix} ⏭️  Skipping "
                    f"({skip_info.get('reason', 'patches exist')})"
                )
                return 0

        logger.info(f"{progress_prefix} Extracting patches...")
        
        # Configure patch extraction.
        # NOTE: map config -> PatchConfig field names (API drift fix):
        #   num_points -> target_num_points (<=0 means "no resampling" -> None)
        #   min_building_points -> kept as building-class filter (LOD2/LOD3 dataset)
        _num_points = OmegaConf.select(
            self.config, "processor.num_points", default=-1
        )
        patch_config = PatchConfig(
            patch_size=self.config.processor.patch_size,
            overlap=OmegaConf.select(
                self.config, "processor.patch_overlap", default=0.1
            ),
            target_num_points=(_num_points if _num_points and _num_points > 0 else None),
            min_points=OmegaConf.select(
                self.config, "processor.min_points", default=10000
            ),
            min_building_points=OmegaConf.select(
                self.config, "processor.min_building_points", default=300
            ),
            building_class=OmegaConf.select(
                self.config, "processor.building_class", default=6
            ),
        )
        
        # PatchConfig carries the augment flags; AugmentationConfig only holds
        # the augmentation hyper-params (its defaults are fine).
        patch_config.augment = self.augment
        patch_config.num_augmentations = self.num_augmentations
        augment_config = AugmentationConfig() if self.augment else None

        # Extract patches
        patches = extract_and_augment_patches(
            points=points,
            features=features,
            labels=classification,
            patch_config=patch_config,
            augment_config=augment_config,
        )

        if len(patches) == 0:
            logger.warning(f"{progress_prefix} ⚠️  No valid patches extracted")
            return 0

        # Save patches
        num_saved = self._save_patches(
            patches=patches,
            laz_file=laz_file,
            output_dir=output_dir,
            architectural_style_id=architectural_style_id,
            multi_styles=multi_styles,
        )

        return num_saved

    def _save_patches(
        self,
        patches: list,
        laz_file: Path,
        output_dir: Path,
        architectural_style_id: int,
        multi_styles: Optional[list],
    ) -> int:
        """
        Save extracted patches in the configured format(s).
        
        Args:
            patches: List of patch dictionaries
            laz_file: Source LAZ file
            output_dir: Output directory
            architectural_style_id: Architectural style ID
            multi_styles: Multi-label styles
            
        Returns:
            Number of patches saved
        """
        # Attach architectural-style metadata (consumed by the writers).
        if self.include_architectural_style:
            for patch in patches:
                patch["architectural_style_id"] = architectural_style_id
                if multi_styles:
                    patch["multi_styles"] = multi_styles

        # Delegate to OutputWriter — the single save path that knows every
        # configured format, including 'ptv3_pointcept' (coord/feat/segment.npy
        # + meta.json). The inline dispatch only handled npz/hdf5/laz/torch/multi,
        # so 'ptv3_pointcept' silently saved 0 patch.
        if self._output_writer is None:
            from .output_writer import OutputWriter
            self._output_writer = OutputWriter(self.config, dataset_manager=None)
        return self._output_writer.save_patches(
            patches=patches,
            laz_file=laz_file,
            output_dir=output_dir,
        )

    def _augment_ground_with_dtm(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment ground points using RGE ALTI DTM (UPGRADED V3.1).

        This function uses the comprehensive DTM augmentation module to add
        synthetic ground points intelligently in areas where they are most needed.

        Args:
            points: Original point cloud [N, 3] (X, Y, Z)
            classification: Point classifications [N]
            bbox: Bounding box (minx, miny, maxx, maxy)

        Returns:
            Tuple of (augmented_points, augmented_classification)
        """
        try:
            from ..io.rge_alti_fetcher import RGEALTIFetcher
            from .classification.dtm_augmentation import (
                AugmentationStrategy,
                DTMAugmentationConfig,
                DTMAugmenter,
            )

            # Get DTM configuration
            dtm_config = OmegaConf.select(
                self.config, "data_sources.rge_alti", default={}
            )
            augment_config = OmegaConf.select(
                self.config, "ground_truth.rge_alti", default={}
            )

            # Initialize RGE ALTI fetcher
            cache_dir = self.config.get("cache_dir")
            if cache_dir:
                cache_dir = Path(cache_dir) / "rge_alti"

            fetcher = RGEALTIFetcher(
                cache_dir=str(cache_dir) if cache_dir else None,
                resolution=dtm_config.get("resolution", 1.0),
                use_wms=dtm_config.get("use_wcs", True),
                api_key=dtm_config.get("api_key", "pratique"),
                prefer_lidar_hd=dtm_config.get("prefer_lidar_hd", True),
            )

            # Map strategy string to enum
            strategy_map = {
                "full": AugmentationStrategy.FULL,
                "gaps": AugmentationStrategy.GAPS,
                "intelligent": AugmentationStrategy.INTELLIGENT,
            }
            strategy_name = augment_config.get("augmentation_strategy", "intelligent")
            strategy = strategy_map.get(strategy_name, AugmentationStrategy.INTELLIGENT)

            # Get priority areas configuration
            priority_config = augment_config.get("augmentation_priority", {})

            # Build augmentation configuration
            aug_config = DTMAugmentationConfig(
                enabled=True,
                strategy=strategy,
                spacing=augment_config.get("augmentation_spacing", 2.0),
                min_spacing_to_existing=augment_config.get("min_spacing_to_existing", 1.5),
                augment_vegetation=priority_config.get("vegetation", True),
                augment_buildings=priority_config.get("buildings", True),
                augment_water=priority_config.get("water", False),
                augment_roads=priority_config.get("roads", False),
                augment_gaps=priority_config.get("gaps", True),
                max_height_difference=augment_config.get("max_height_difference", 5.0),
                validate_against_neighbors=augment_config.get("validate_against_neighbors", True),
                min_neighbors_for_validation=augment_config.get("min_neighbors_for_validation", 3),
                neighbor_search_radius=augment_config.get("neighbor_search_radius", 10.0),
                synthetic_ground_class=augment_config.get("synthetic_ground_class", 2),
                mark_as_synthetic=augment_config.get("mark_as_synthetic", True),
                verbose=True,
            )

            # Get building polygons if available
            building_polygons = None
            if self.data_fetcher is not None:
                try:
                    minx, miny, maxx, maxy = bbox
                    building_gdf = self.data_fetcher._fetch_bd_topo_buildings(minx, miny, maxx, maxy)
                    if building_gdf is not None and len(building_gdf) > 0:
                        building_polygons = building_gdf
                        logger.debug(f"      Using {len(building_polygons)} building polygons for targeted augmentation")
                except Exception as e:
                    logger.debug(f"      Could not fetch building polygons: {e}")

            # Create augmenter and run augmentation
            augmenter = DTMAugmenter(config=aug_config)
            augmented_points, augmented_labels, augmentation_attrs = augmenter.augment_point_cloud(
                points=points,
                labels=classification,
                dtm_fetcher=fetcher,
                bbox=bbox,
                building_polygons=building_polygons,
                crs="EPSG:2154",
            )

            # Check if any points were added
            n_added = len(augmented_points) - len(points)
            if n_added > 0:
                # Store augmentation stats
                if augment_config.get("save_augmentation_report", True):
                    self._store_augmentation_stats(augmentation_attrs, n_added)
                return augmented_points, augmented_labels
            else:
                logger.info("      ℹ️  No ground points added (sufficient existing coverage)")
                return points, classification

        except ImportError as e:
            logger.error(f"      ❌ DTM augmentation module not available: {e}")
            return points, classification
        except Exception as e:
            logger.error(f"      ❌ Ground augmentation failed: {e}")
            logger.debug("      Exception details:", exc_info=True)
            return points, classification

    def _store_augmentation_stats(self, augmentation_attrs: dict, n_added: int):
        """Store DTM augmentation statistics for later reporting."""
        if not hasattr(self, "_augmentation_stats"):
            self._augmentation_stats = []
        
        self._augmentation_stats.append({
            "n_added": n_added,
            "attributes": augmentation_attrs
        })
                
        return num_saved
