"""
Patch extraction and augmentation for machine learning datasets.

This module handles extracting fixed-size patches from processed point clouds
and applying data augmentation for training dataset diversity.

Extracted from LiDARProcessor as part of refactoring (v3.4.0).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from .classification.patch_extractor import (
    AugmentationConfig,
    PatchConfig,
    extract_and_augment_patches,
)

logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extract and augment patches from point clouds for ML training.

    Handles:
    - Patch extraction with configurable size and overlap
    - Data augmentation (rotation, jitter, scaling)
    - Architecture-specific formatting (PointNet++, DGCNN, etc.)
    - Class-based patch filtering
    - Patch validation

    This class was extracted from LiDARProcessor to separate patch
    generation concerns from tile processing.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize patch extractor with configuration.

        Args:
            config: Configuration containing patch extraction parameters
        """
        self.config = config
        self.patch_size = config.processor.patch_size
        self.num_points = config.processor.num_points
        self.architecture = config.processor.architecture
        self.augment = config.processor.get("augment", False)
        self.num_augmentations = config.processor.get("num_augmentations", 3)
        self.patch_overlap = config.processor.get("patch_overlap", 0.1)

        # Create patch configuration
        self.patch_config = PatchConfig(
            patch_size=self.patch_size,
            target_num_points=self.num_points,  # Correct parameter name
            overlap=self.patch_overlap,
            min_points=config.processor.get("min_points_per_patch", 10000),
            augment=self.augment,
            num_augmentations=self.num_augmentations,
        )

        # Create augmentation configuration if enabled
        self.aug_config = None
        if self.augment:
            self.aug_config = AugmentationConfig(
                rotation_range=config.processor.get("rotation_range", 2 * np.pi),
                jitter_sigma=config.processor.get("jitter_sigma", 0.01),
                scale_range=config.processor.get("scale_range", (0.95, 1.05)),
                dropout_range=config.processor.get("dropout_range", (0.05, 0.15)),
                apply_to_raw_points=config.processor.get("apply_to_raw_points", True),
            )

        logger.debug(
            f"Initialized PatchExtractor: "
            f"patch_size={self.patch_size}m, "
            f"num_points={self.num_points}, "
            f"augment={self.augment}"
        )

    def extract_patches(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract patches from processed point cloud.

        Args:
            points: Point cloud array [N, 3] with XYZ coordinates
            features: Dictionary of computed features
            classification: Classification codes [N]
            metadata: Optional tile metadata

        Returns:
            List of patch dictionaries, each containing:
            - 'points': Patch point cloud [M, 3]
            - 'features': Patch features dict
            - 'classification': Patch classifications [M]
            - '_version': Patch version (original or augmented)
            - '_patch_idx': Patch index

        Note:
            Patches are extracted using the classification module's
            extract_and_augment_patches function, which handles:
            - Grid-based extraction with overlap
            - Point count validation
            - Data augmentation (if enabled)
            - Architecture-specific formatting
        """
        # Extract patches using the classification module's function
        patches = extract_and_augment_patches(
            points=points,
            features=features,
            labels=classification,
            patch_config=self.patch_config,
            augment_config=self.aug_config if self.augment else None,
            architecture=self.architecture,
            source_crs=(metadata or {}).get("crs", "EPSG:2154"),
            source_bbox=(metadata or {}).get(
                "source_bbox", (metadata or {}).get("bounds")
            ),
            source_indices=(metadata or {}).get("source_indices"),
            logger_instance=logger,
        )

        logger.info(
            f"Extracted {len(patches)} patches "
            f"(patch_size={self.patch_size}m, "
            f"target_points={self.num_points})"
        )

        return patches

    def extract_patches_by_class(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        classification: np.ndarray,
        target_classes: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, List[Dict[str, np.ndarray]]]:
        """
        Extract patches grouped by classification.

        Useful for creating class-balanced datasets or analyzing
        specific object types.

        Args:
            points: Point cloud array [N, 3]
            features: Feature dictionary
            classification: Classification codes [N]
            target_classes: List of classes to extract
            metadata: Optional tile metadata

        Returns:
            Dictionary mapping class code to list of patches

        Example:
            >>> # Extract building and ground patches separately
            >>> patches_by_class = extractor.extract_patches_by_class(
            ...     points, features, classification,
            ...     target_classes=[2, 6]  # Ground=2, Building=6
            ... )
            >>> building_patches = patches_by_class[6]
            >>> ground_patches = patches_by_class[2]
        """
        patches_by_class = {cls: [] for cls in target_classes}

        # Extract all patches
        all_patches = self.extract_patches(points, features, classification, metadata)

        # Group by dominant class
        for patch in all_patches:
            patch_classification = patch["classification"]
            dominant_class = int(np.bincount(patch_classification).argmax())

            if dominant_class in target_classes:
                patches_by_class[dominant_class].append(patch)

        # Log statistics
        for cls, patches in patches_by_class.items():
            logger.info(f"Class {cls}: {len(patches)} patches extracted")

        return patches_by_class

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get patch extractor statistics.

        Returns:
            Dictionary with extraction configuration and stats
        """
        return {
            "patch_size": self.patch_size,
            "num_points": self.num_points,
            "architecture": self.architecture,
            "augmentation_enabled": self.augment,
            "num_augmentations": self.num_augmentations,
            "patch_overlap": self.patch_overlap,
        }
