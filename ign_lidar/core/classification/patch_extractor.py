"""
Patch Extraction and Augmentation Module

This module handles patch extraction and data augmentation for LiDAR point clouds:
- Grid-based patch extraction with overlap
- Point cloud resampling (up/down) to target size
- Data augmentation (rotation, jitter, scaling, dropout)
- Architecture-specific patch formatting
- Multi-architecture support

Extracted from processor.py and preprocessing/utils.py as part of Phase 4 refactoring.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class PatchConfig:
    """Configuration for patch extraction.
    
    Attributes:
        patch_size: Patch size in meters
        overlap: Overlap ratio between patches (0-1)
        min_points: Minimum points required per patch
        target_num_points: Target number of points per patch (None = no resampling)
        augment: Whether to create augmented versions
        num_augmentations: Number of augmented versions per patch
    """
    patch_size: float = 150.0
    overlap: float = 0.1
    min_points: int = 10000
    target_num_points: Optional[int] = None
    min_building_points: int = 0       # 0 = disabled ; >0 keeps only patches with
    building_class: int = 6            # >= N points of this class (ASPRS 6 = building)
    augment: bool = False
    num_augmentations: int = 3


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.
    
    Attributes:
        rotation_range: Rotation range in radians (0 to this value)
        jitter_sigma: Standard deviation for Gaussian jitter
        scale_range: Tuple of (min_scale, max_scale)
        dropout_range: Tuple of (min_dropout, max_dropout) ratios
        apply_to_raw_points: Whether to augment before feature computation
    """
    rotation_range: float = 2 * np.pi  # 0-360 degrees
    jitter_sigma: float = 0.1  # meters (for raw) or 0.01 (for normalized patches)
    scale_range: Tuple[float, float] = (0.95, 1.05)
    dropout_range: Tuple[float, float] = (0.05, 0.15)
    apply_to_raw_points: bool = True


# ============================================================================
# Patch Extraction Functions
# ============================================================================

def extract_patches(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    patch_size: float = 150.0,
    overlap: float = 0.1,
    min_points: int = 10000,
    target_num_points: Optional[int] = None,
    min_building_points: int = 0,
    building_class: int = 6,
    source_crs: Optional[str] = "EPSG:2154",
    source_bbox: Optional[Sequence[float]] = None,
    source_indices: Optional[np.ndarray] = None,
    logger_instance: Optional[logging.Logger] = None
) -> List[Dict[str, np.ndarray]]:
    """Extract patches from point cloud with overlap.
    
    Uses a grid-based approach with configurable overlap. Each patch is centered
    at its grid position and contains all points within the patch bounds.
    
    Args:
        points: Point cloud [N, 3] with XYZ coordinates
        features: Dictionary of point features (intensity, normals, etc.)
        labels: Point labels/classifications [N]
        patch_size: Patch size in meters
        overlap: Overlap ratio between patches (0-1)
        min_points: Minimum points required per patch
        target_num_points: Target number of points (None = no resampling)
        source_crs: CRS of ``points``. IGN LiDAR HD defaults to Lambert-93.
        source_bbox: Optional source extent as
            ``(xmin, ymin, zmin, xmax, ymax, zmax)``. Computed from ``points``
            when omitted.
        source_indices: Optional indices mapping ``points`` back to the source
            point cloud. They are filtered/resampled together with the patch.
        logger_instance: Optional logger instance
        
    Returns:
        List of patch dictionaries, each containing:
        - 'points': Centered point coordinates [M, 3]
        - 'labels': Point labels [M]
        - '_patch_center': Original center coordinates in tile space [3]
        - '_patch_bounds': Patch bounds (x_start, y_start, x_end, y_end)
        - '_source_bbox': Source point-cloud extent [6]
        - '_crs': Source CRS identifier
        - 'source_indices': Optional point-wise indices into the source cloud
        - Additional features from features dict
    """
    log = logger_instance or logger

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if len(points) == 0:
        return []
    if source_indices is not None:
        source_indices = np.asarray(source_indices)
        if source_indices.ndim != 1 or len(source_indices) != len(points):
            raise ValueError(
                "source_indices must be a 1-D array with one entry per point"
            )

    if source_bbox is None:
        source_min = points.astype(np.float64, copy=False).min(axis=0)
        source_max = points.astype(np.float64, copy=False).max(axis=0)
        source_bbox_array = np.concatenate((source_min, source_max))
    else:
        source_bbox_array = np.asarray(source_bbox, dtype=np.float64).ravel()
        if source_bbox_array.shape != (6,):
            raise ValueError(
                "source_bbox must be (xmin, ymin, zmin, xmax, ymax, zmax)"
            )
    if not np.isfinite(source_bbox_array).all():
        raise ValueError("source_bbox contains NaN or infinite values")
    
    # Compute bounding box
    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)
    
    # Compute patch grid with overlap
    stride = patch_size * (1 - overlap)
    x_steps = int(np.ceil((x_max - x_min) / stride))
    y_steps = int(np.ceil((y_max - y_min) / stride))
    
    log.debug(f"Patch grid: {x_steps}x{y_steps} = {x_steps * y_steps} potential patches")
    
    patches = []
    
    for i in range(x_steps):
        for j in range(y_steps):
            # Define patch bounds
            x_start = x_min + i * stride
            y_start = y_min + j * stride
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Find points in patch
            mask = (
                (points[:, 0] >= x_start) & (points[:, 0] < x_end) &
                (points[:, 1] >= y_start) & (points[:, 1] < y_end)
            )
            
            num_points_in_patch = np.sum(mask)
            if num_points_in_patch < min_points:
                continue

            # Extract patch data
            patch_points = points[mask]
            patch_labels = labels[mask]

            # Building-oriented filter: keep the patch only if it holds enough
            # points of the building class (LOD2/LOD3 dataset). Disabled if 0.
            if min_building_points > 0:
                n_building = int(np.count_nonzero(patch_labels == building_class))
                if n_building < min_building_points:
                    continue
            
            # Center patch coordinates
            patch_center = np.array([
                (x_start + x_end) / 2,
                (y_start + y_end) / 2,
                0
            ])
            patch_points = patch_points - patch_center
            
            # Build patch dictionary with metadata
            patch = {
                'points': patch_points,
                'labels': patch_labels,
                '_patch_center': patch_center,  # Store for debugging/validation
                '_patch_bounds': (x_start, y_start, x_end, y_end),  # Store bounds
                '_source_bbox': source_bbox_array.copy(),
                '_crs': source_crs,
            }
            if source_indices is not None:
                patch['source_indices'] = source_indices[mask]
            
            # Add features
            for feature_name, feature_data in features.items():
                patch[feature_name] = feature_data[mask]
            
            # Resample to target number of points if specified
            if target_num_points is not None:
                patch = resample_patch(patch, target_num_points)
            
            patches.append(patch)
    
    log.debug(f"Extracted {len(patches)} patches (>= {min_points} points)")
    return patches


def resample_patch(
    patch: Dict[str, np.ndarray],
    target_num_points: int
) -> Dict[str, np.ndarray]:
    """Resample patch to target number of points.
    
    - If patch has more points: random downsampling
    - If patch has fewer points: random upsampling with replacement
    - If patch has exact number: no change
    
    Args:
        patch: Input patch dictionary
        target_num_points: Target number of points
        
    Returns:
        Resampled patch dictionary
    """
    num_points_current = len(patch['points'])
    
    if num_points_current == target_num_points:
        return patch
    
    if num_points_current > target_num_points:
        # Downsample: random selection without replacement
        indices = np.random.choice(
            num_points_current,
            target_num_points,
            replace=False
        )
    else:
        # Upsample: random selection with replacement
        indices = np.random.choice(
            num_points_current,
            target_num_points,
            replace=True
        )
    
    # Apply resampling to point-wise arrays (not metadata)
    # Metadata fields start with '_' and should not be resampled
    resampled_patch = {}
    for key, value in patch.items():
        if key.startswith('_'):
            # Metadata field - copy as-is
            resampled_patch[key] = value
        elif isinstance(value, np.ndarray) and len(value) == num_points_current:
            # Point-wise feature array - resample
            resampled_patch[key] = value[indices]
        else:
            # Other data - copy as-is
            resampled_patch[key] = value
    
    return resampled_patch


# ============================================================================
# Augmentation Functions
# ============================================================================

def augment_raw_points(
    points: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    classification: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
    ndvi: Optional[np.ndarray] = None,
    config: Optional[AugmentationConfig] = None,
    return_mask: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
          Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]
]:
    """Apply data augmentation to raw point cloud data BEFORE feature computation.
    
    This ensures geometric features (normals, curvature, planarity, etc.) are
    computed on the augmented geometry, maintaining consistency between
    coordinates and derived features.
    
    Augmentations applied:
    1. Random rotation around Z-axis
    2. Random jitter (Gaussian noise)
    3. Random scaling
    4. Random point dropout
    
    Args:
        points: Raw point coordinates [N, 3]
        intensity: Intensity values [N]
        return_number: Return numbers [N]
        classification: ASPRS classification codes [N]
        rgb: RGB values [N, 3] (optional)
        nir: Near-infrared values [N] (optional)
        ndvi: NDVI values [N] (optional)
        config: Augmentation configuration (None = use defaults)
        return_mask: If True, return keep_mask as last element
        
    Returns:
        Tuple of (augmented_points, intensity, return_number, classification, rgb, nir, ndvi)
        If return_mask=True, also returns keep_mask as 8th element
    """
    if config is None:
        config = AugmentationConfig()
    
    N = len(points)
    points_aug = points.copy()
    
    # 1. Random rotation around Z-axis (preserves vertical structures)
    angle = np.random.uniform(0, config.rotation_range)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    points_aug = points_aug @ rotation_matrix.T
    
    # 2. Random jitter (simulates sensor noise)
    jitter = np.random.normal(0, config.jitter_sigma, (N, 3)).astype(np.float32)
    points_aug += jitter
    
    # 3. Random scaling (simulates distance variations)
    scale = np.random.uniform(config.scale_range[0], config.scale_range[1])
    points_aug *= scale
    
    # 4. Random dropout (simulates occlusion, missing data)
    dropout_ratio = np.random.uniform(config.dropout_range[0], config.dropout_range[1])
    keep_mask = np.random.random(N) > dropout_ratio
    
    # Apply dropout to all arrays
    points_aug = points_aug[keep_mask]
    intensity_aug = intensity[keep_mask]
    return_number_aug = return_number[keep_mask]
    classification_aug = classification[keep_mask]
    
    # Apply dropout to optional arrays
    rgb_aug = rgb[keep_mask] if rgb is not None else None
    nir_aug = nir[keep_mask] if nir is not None else None
    ndvi_aug = ndvi[keep_mask] if ndvi is not None else None
    
    if return_mask:
        return (points_aug, intensity_aug, return_number_aug, classification_aug,
                rgb_aug, nir_aug, ndvi_aug, keep_mask)
    else:
        return (points_aug, intensity_aug, return_number_aug, classification_aug,
                rgb_aug, nir_aug, ndvi_aug)


def augment_patch(
    patch: Dict[str, np.ndarray],
    config: Optional[AugmentationConfig] = None
) -> Dict[str, np.ndarray]:
    """Apply data augmentation to a single patch.
    
    Applies geometric transformations to a patch AFTER it has been extracted,
    ensuring augmented versions correspond to the same spatial region.
    
    Uses smaller jitter (σ=0.01) for normalized patches.
    
    Args:
        patch: Input patch dictionary containing:
            - 'points': Point coordinates [N, 3] (normalized/centered)
            - 'labels': Classification labels [N]
            - 'intensity': Intensity values [N]
            - 'return_number': Return numbers [N]
            - Additional features (optional)
        config: Augmentation configuration (None = use defaults with σ=0.01)
        
    Returns:
        Augmented patch dictionary with same keys as input
    """
    if config is None:
        config = AugmentationConfig(jitter_sigma=0.01)  # Smaller jitter for normalized patches
    
    aug_patch = {}
    
    # Copy points and get dimensions
    points = patch['points'].copy()
    N = len(points)
    
    # 1. Random rotation around Z-axis
    angle = np.random.uniform(0, config.rotation_range)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    points = points @ rotation_matrix.T
    
    # 2. Random jitter (smaller for normalized patches)
    jitter = np.random.normal(0, config.jitter_sigma, (N, 3)).astype(np.float32)
    points += jitter
    
    # 3. Random scaling
    scale = np.random.uniform(config.scale_range[0], config.scale_range[1])
    points *= scale
    
    # 4. Random dropout
    dropout_ratio = np.random.uniform(config.dropout_range[0], config.dropout_range[1])
    keep_mask = np.random.random(N) > dropout_ratio
    
    # Apply transformations
    aug_patch['points'] = points[keep_mask]
    
    # Apply dropout to all other features
    for key, value in patch.items():
        if key == 'points':
            continue
        # Skip metadata fields (start with _)
        if key.startswith('_'):
            aug_patch[key] = value
            continue
        if isinstance(value, np.ndarray):
            aug_patch[key] = value[keep_mask]
        else:
            aug_patch[key] = value
    
    # Rotate normals if present (normals need rotation too)
    if 'normals' in aug_patch:
        normals = aug_patch['normals']
        aug_patch['normals'] = normals @ rotation_matrix.T
    
    return aug_patch


# ============================================================================
# Multi-Version Patch Creation
# ============================================================================

def create_patch_versions(
    base_patches: List[Dict[str, np.ndarray]],
    num_augmentations: int = 3,
    augment_config: Optional[AugmentationConfig] = None,
    logger_instance: Optional[logging.Logger] = None
) -> List[Dict[str, np.ndarray]]:
    """Create multiple versions of patches (original + augmented).
    
    For each base patch:
    1. Add original patch with metadata {'_version': 'original', '_patch_idx': idx}
    2. Create N augmented versions with metadata {'_version': 'aug_0', ..., '_patch_idx': idx}
    
    Args:
        base_patches: List of base patch dictionaries
        num_augmentations: Number of augmented versions per patch
        augment_config: Augmentation configuration
        logger_instance: Optional logger instance
        
    Returns:
        List of all patch versions (original + augmented)
    """
    log = logger_instance or logger
    
    all_patches = []
    
    for patch_idx, base_patch in enumerate(base_patches):
        # Version 0: Original patch
        original_patch = base_patch.copy()
        original_patch['_version'] = 'original'
        original_patch['_patch_idx'] = patch_idx
        all_patches.append(original_patch)
        
        # Augmented versions
        for aug_idx in range(num_augmentations):
            aug_patch = augment_patch(base_patch, config=augment_config)
            aug_patch['_version'] = f'aug_{aug_idx}'
            aug_patch['_patch_idx'] = patch_idx
            all_patches.append(aug_patch)
    
    num_versions = 1 + num_augmentations
    log.info(
        f"  🔄 Created {len(all_patches)} total patches "
        f"({len(base_patches)} base x {num_versions} versions)"
    )
    
    return all_patches


# ============================================================================
# Architecture-Specific Formatting
# ============================================================================

def format_patch_for_architecture(
    patch: Dict[str, np.ndarray],
    architecture: str,
    num_points: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Format patch for specific neural network architecture.
    
    Different architectures expect different input formats:
    - PointNet/PointNet++: [N, C] where C = XYZ + features
    - DGCNN: [N, C] similar to PointNet
    - RandLA-Net: [N, C] + neighbor indices
    - KPConv: [N, C] + subsampling/upsampling indices
    
    Currently handles basic formatting. Can be extended for architecture-specific needs.
    
    Args:
        patch: Input patch dictionary
        architecture: Architecture name ('pointnet', 'pointnet2', 'dgcnn', etc.)
        num_points: Target number of points (None = keep as is)
        
    Returns:
        Formatted patch dictionary
    """
    formatted_patch = patch.copy()
    
    # Resample if needed
    if num_points is not None and len(patch['points']) != num_points:
        formatted_patch = resample_patch(formatted_patch, num_points)
    
    # Architecture-specific formatting can be added here
    if architecture.lower() in ['pointnet', 'pointnet2', 'pointnet++']:
        # PointNet expects [N, C] tensor
        # Already in correct format
        pass
    
    elif architecture.lower() == 'dgcnn':
        # DGCNN expects [N, C] tensor
        # Already in correct format
        pass
    
    elif architecture.lower() in ['randla', 'randlanet', 'randla-net']:
        # RandLA-Net may need additional preprocessing
        # Typically handled by architecture-specific dataloader
        pass
    
    elif architecture.lower() in ['kpconv', 'kp-conv']:
        # KPConv may need subsampling info
        # Typically handled by architecture-specific dataloader
        pass
    
    return formatted_patch


# ============================================================================
# High-Level Extraction Pipeline
# ============================================================================

def extract_and_augment_patches(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    patch_config: PatchConfig,
    augment_config: Optional[AugmentationConfig] = None,
    architecture: Optional[str] = None,
    source_crs: Optional[str] = "EPSG:2154",
    source_bbox: Optional[Sequence[float]] = None,
    source_indices: Optional[np.ndarray] = None,
    logger_instance: Optional[logging.Logger] = None
) -> List[Dict[str, np.ndarray]]:
    """Complete pipeline: extract patches + create augmented versions.
    
    This is the main high-level function that combines:
    1. Patch extraction from point cloud
    2. Creating augmented versions (if enabled)
    3. Architecture-specific formatting (if specified)
    
    Args:
        points: Point cloud [N, 3]
        features: Dictionary of point features
        labels: Point labels [N]
        patch_config: Patch extraction configuration
        augment_config: Augmentation configuration
        architecture: Target architecture name (optional)
        source_crs: CRS of the source point cloud.
        source_bbox: Optional full source extent [6].
        source_indices: Optional indices into the source point cloud.
        logger_instance: Optional logger instance
        
    Returns:
        List of patch dictionaries (original + augmented versions)
    """
    log = logger_instance or logger
    
    # Step 1: Extract base patches
    log.info(
        f"  📦 Extracting patches (size={patch_config.patch_size}m, "
        f"target_points={patch_config.target_num_points})..."
    )
    
    base_patches = extract_patches(
        points=points,
        features=features,
        labels=labels,
        patch_size=patch_config.patch_size,
        overlap=patch_config.overlap,
        min_points=patch_config.min_points,
        target_num_points=patch_config.target_num_points,
        min_building_points=patch_config.min_building_points,
        building_class=patch_config.building_class,
        source_crs=source_crs,
        source_bbox=source_bbox,
        source_indices=source_indices,
        logger_instance=log
    )
    
    log.info(f"  ✓ Extracted {len(base_patches)} base patches")
    
    # Step 2: Create augmented versions if enabled
    if patch_config.augment:
        all_patches = create_patch_versions(
            base_patches=base_patches,
            num_augmentations=patch_config.num_augmentations,
            augment_config=augment_config,
            logger_instance=log
        )
    else:
        # No augmentation: just add metadata to base patches
        all_patches = []
        for idx, patch in enumerate(base_patches):
            patch['_version'] = 'original'
            patch['_patch_idx'] = idx
            all_patches.append(patch)
        log.info(f"  ✓ Using {len(all_patches)} patches (no augmentation)")
    
    # Step 3: Architecture-specific formatting if specified
    if architecture is not None:
        log.debug(f"  🏗️  Formatting patches for {architecture}")
        all_patches = [
            format_patch_for_architecture(
                patch,
                architecture,
                patch_config.target_num_points
            )
            for patch in all_patches
        ]
    
    return all_patches


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Configuration
    'PatchConfig',
    'AugmentationConfig',
    
    # Extraction functions
    'extract_patches',
    'resample_patch',
    
    # Augmentation functions
    'augment_raw_points',
    'augment_patch',
    'create_patch_versions',
    
    # Formatting
    'format_patch_for_architecture',
    
    # High-level pipeline
    'extract_and_augment_patches',
]
