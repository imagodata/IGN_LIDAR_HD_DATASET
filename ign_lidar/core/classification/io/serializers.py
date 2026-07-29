"""
Serialization module for saving patches and enriched data in various formats.

This module provides functions to save LiDAR patches and enriched point clouds
in multiple output formats: NPZ, HDF5, PyTorch, and LAZ.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.debug("h5py not available, HDF5 format not supported")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("torch not available, PyTorch format not supported")

try:
    import laspy

    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.debug("laspy not available, LAZ format not supported")


def save_patch_npz(
    save_path: Path, data: Dict[str, np.ndarray], lod_level: Optional[str] = None
) -> None:
    """
    Save patch to compressed NPZ file.

    Args:
        save_path: Output NPZ file path
        data: Patch data dictionary with numpy arrays
        lod_level: Optional LOD level string to include in saved data
    """
    save_data = data.copy()
    if lod_level is not None:
        save_data["lod_level"] = lod_level

    np.savez_compressed(save_path, **save_data)
    logger.debug(f"Saved NPZ patch: {save_path.name}")


def save_patch_hdf5(save_path: Path, data: Dict[str, np.ndarray]) -> None:
    """
    Save patch to HDF5 file with GZIP compression.

    Args:
        save_path: Output HDF5 file path
        data: Patch data dictionary

    Raises:
        ImportError: If h5py is not installed
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 format. " "Install with: pip install h5py"
        )

    with h5py.File(save_path, "w") as f:
        # Extract metadata (don't modify data - use get instead of pop)
        metadata = data.get("metadata", None)

        # Save all numpy arrays as datasets
        for key, value in data.items():
            # Skip metadata and non-array fields
            if key == "metadata":
                continue
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key, data=value, compression="gzip", compression_opts=9
                )
            else:
                logger.warning(
                    f"Skipping non-array field '{key}' for HDF5 "
                    f"(type: {type(value).__name__})"
                )

        # Save metadata as HDF5 attributes (flattened)
        if metadata is not None:
            for meta_key, meta_value in metadata.items():
                try:
                    # Convert lists to arrays for HDF5 compatibility
                    if isinstance(meta_value, list):
                        meta_value = np.array(meta_value)
                    f.attrs[meta_key] = meta_value
                except (TypeError, ValueError) as e:
                    logger.debug(f"Could not save metadata key '{meta_key}': {e}")

    logger.debug(f"Saved HDF5 patch: {save_path.name}")


def save_patch_torch(save_path: Path, data: Dict[str, np.ndarray]) -> None:
    """
    Save patch to PyTorch .pt file.

    Converts numpy arrays to torch tensors before saving.

    Args:
        save_path: Output PyTorch file path
        data: Patch data dictionary

    Raises:
        ImportError: If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for torch format. " "Install with: pip install torch"
        )

    # Convert numpy arrays to torch tensors (skip metadata and non-arrays)
    torch_data = {}
    for k, v in data.items():
        if k == "metadata":
            # Keep metadata as-is for PyTorch (it supports dicts)
            torch_data[k] = v
        elif isinstance(v, np.ndarray):
            torch_data[k] = torch.from_numpy(v)
        else:
            logger.warning(
                f"Skipping non-array field '{k}' for PyTorch "
                f"(type: {type(v).__name__})"
            )

    torch.save(torch_data, save_path)
    logger.debug(f"Saved PyTorch patch: {save_path.name}")


def save_patch_laz(
    save_path: Path,
    arch_data: Dict[str, np.ndarray],
    original_patch: Dict[str, np.ndarray],
) -> None:
    """
    Save a patch as a LAZ point cloud file with all computed features.

    This function creates a LAZ file with coordinates, classification, intensity,
    RGB colors (if available), NIR (if available), and adds all geometric and
    radiometric features as extra dimensions.

    Args:
        save_path: Output LAZ file path
        arch_data: Formatted patch data (architecture-specific)
        original_patch: Original patch data with metadata and features

    Raises:
        ImportError: If laspy is not installed
    """
    if not LASPY_AVAILABLE:
        raise ImportError(
            "laspy is required for LAZ format. " "Install with: pip install laspy"
        )

    # Extract coordinates from arch_data
    # Most architectures have 'points' or 'coords' key
    if "points" in arch_data:
        coords = arch_data["points"][:, :3].copy()  # XYZ
    elif "coords" in arch_data:
        coords = arch_data["coords"][:, :3].copy()
    else:
        logger.warning(
            f"Cannot save LAZ patch: no coordinates found in "
            f"{list(arch_data.keys())}"
        )
        return

    # Restore LAMB93 coordinates if metadata available
    coords = _restore_lamb93_coordinates(save_path, coords, original_patch)

    # Determine point format based on available data
    has_rgb = "rgb" in arch_data
    has_nir = "nir" in original_patch and original_patch["nir"] is not None
    point_format = 8 if (has_rgb and has_nir) else (3 if has_rgb else 6)

    # Create LAZ file
    header = laspy.LasHeader(version="1.4", point_format=point_format)
    header.offsets = [
        np.floor(coords[:, 0].min()),
        np.floor(coords[:, 1].min()),
        np.floor(coords[:, 2].min()),
    ]
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]

    # Add standard LAS fields
    _add_standard_las_fields(las, arch_data, original_patch, point_format)

    # Track added extra dimensions to avoid duplicates
    added_dimensions = set()

    # Add computed features as extra dimensions
    _add_geometric_features(las, original_patch, added_dimensions)
    _add_height_features(las, original_patch, added_dimensions)
    _add_radiometric_features(las, original_patch, point_format, added_dimensions)
    _add_return_number(las, original_patch, added_dimensions)

    # Write LAZ file
    las.write(str(save_path))
    logger.debug(f"Saved LAZ patch: {save_path.name}")


def _restore_lamb93_coordinates(
    save_path: Path, coords: np.ndarray, original_patch: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Restore LAMB93 coordinates from normalized patch coordinates.

    Args:
        save_path: Output file path (filename contains tile coordinates)
        coords: Normalized coordinates [N, 3]
        original_patch: Original patch with metadata

    Returns:
        Restored coordinates in LAMB93 projection
    """
    try:
        filename = save_path.stem
        parts = filename.split("_")

        # Look for tile coordinates in filename (e.g., LHD_FXX_0649_6863_...)
        if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
            tile_x = int(parts[2])
            tile_y = int(parts[3])

            # LAMB93 tiles are 1km x 1km
            # tile center at (tile_x * 1000 + 500, tile_y * 1000 + 500)
            tile_center_x = tile_x * 1000 + 500
            tile_center_y = tile_y * 1000 + 500

            # If metadata has centroid, restore original coordinates
            if "metadata" in original_patch and isinstance(
                original_patch["metadata"], dict
            ):
                metadata = original_patch["metadata"]
                if "centroid" in metadata:
                    centroid = np.array(metadata["centroid"])
                    coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
                    coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
                    coords[:, 2] = coords[:, 2] + centroid[2]
                else:
                    # No centroid, just apply tile offset
                    coords[:, 0] = coords[:, 0] + tile_center_x
                    coords[:, 1] = coords[:, 1] + tile_center_y
    except Exception as e:
        logger.debug(f"Could not restore LAMB93 coordinates: {e}")
        # Continue with normalized coordinates

    return coords


def _add_standard_las_fields(
    las, arch_data: Dict, original_patch: Dict, point_format: int
) -> None:
    """Add standard LAS fields (classification, intensity, RGB, NIR)."""
    # Add classification if available
    if "labels" in arch_data:
        las.classification = arch_data["labels"].astype(np.uint8)
    elif "classification" in original_patch:
        las.classification = original_patch["classification"].astype(np.uint8)

    # Add intensity if available
    if "intensity" in original_patch:
        las.intensity = (original_patch["intensity"] * 65535).astype(np.uint16)

    # Add RGB if available
    if "rgb" in arch_data:
        rgb = (arch_data["rgb"] * 65535).astype(np.uint16)
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    # Add NIR if available and point format supports it (format 8)
    if (
        point_format == 8
        and "nir" in original_patch
        and original_patch["nir"] is not None
    ):
        nir = (original_patch["nir"] * 65535).astype(np.uint16)
        las.nir = nir


def _add_geometric_features(las, original_patch: Dict, added_dimensions: set) -> None:
    """Add ALL geometric features as extra dimensions (supports full mode with 40+ features)."""
    # ALL geometric features in consistent order (matches base_formatter.py)
    geometric_features = [
        # Core shape descriptors (6)
        "planarity",
        "linearity",
        "sphericity",
        "anisotropy",
        "roughness",
        "omnivariance",
        # Curvature features (2)
        "curvature",
        "change_curvature",
        # Eigenvalue features (5)
        "eigenvalue_1",
        "eigenvalue_2",
        "eigenvalue_3",
        "sum_eigenvalues",
        "eigenentropy",
        # Building scores (4)
        "verticality",
        "horizontality",
        "wall_score",
        "roof_score",
        # Density features (5)
        "density",
        "local_density",
        "num_points_2m",
        "neighborhood_extent",
        "height_extent_ratio",
        # Architectural features (5)
        "edge_strength",
        "corner_likelihood",
        "overhang_indicator",
        "surface_roughness",
        "local_roughness",
    ]

    for feat_name in geometric_features:
        if feat_name in original_patch and feat_name not in added_dimensions:
            try:
                # Truncate description to 31 chars max (LAZ limit - needs null terminator)
                desc = f"Geom: {feat_name}"[:31]
                las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name=feat_name, type=np.float32, description=desc
                    )
                )
                setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
                added_dimensions.add(feat_name)
            except Exception as e:
                logger.warning(f"Could not add feature '{feat_name}' to LAZ: {e}")

    # Normals (normal_x, normal_y, normal_z)
    if "normals" in original_patch:
        try:
            normals = original_patch["normals"]
            for i, comp in enumerate(["normal_x", "normal_y", "normal_z"]):
                if comp not in added_dimensions:
                    las.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name=comp, type=np.float32, description=f"Normal {comp[-1]}"
                        )
                    )
                    setattr(las, comp, normals[:, i].astype(np.float32))
                    added_dimensions.add(comp)
        except Exception as e:
            logger.warning(f"Could not add normals to LAZ: {e}")


def _add_height_features(las, original_patch: Dict, added_dimensions: set) -> None:
    """Add ALL height features as extra dimensions."""
    height_features = [
        # Core height features
        "height",
        "height_above_ground",
        "vertical_std",
        # Normalized height features
        "z_normalized",
        "z_absolute",
        "z_from_ground",
        "z_from_median",
        # Distance features
        "distance_to_center",
    ]

    for feat_name in height_features:
        if feat_name in original_patch and feat_name not in added_dimensions:
            try:
                # Truncate description to 31 chars max (LAZ limit - needs null terminator)
                desc = f"Height: {feat_name}"[:31]
                las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name=feat_name, type=np.float32, description=desc
                    )
                )
                setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
                added_dimensions.add(feat_name)
            except Exception as e:
                logger.warning(f"Could not add feature '{feat_name}' to LAZ: {e}")


def _add_radiometric_features(
    las, original_patch: Dict, point_format: int, added_dimensions: set
) -> None:
    """Add radiometric features (NIR, NDVI) as extra dimensions."""
    # Add NIR as extra dimension if not already included in point format
    if (
        point_format != 8
        and "nir" in original_patch
        and original_patch["nir"] is not None
    ):
        if "nir" not in added_dimensions:
            try:
                las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="nir",
                        type=np.float32,
                        description="NIR reflectance (norm 0-1)",
                    )
                )
                las.nir = original_patch["nir"].astype(np.float32)
                added_dimensions.add("nir")
            except Exception as e:
                logger.warning(f"Could not add NIR to LAZ: {e}")

    if "ndvi" in original_patch and original_patch["ndvi"] is not None:
        if "ndvi" not in added_dimensions:
            try:
                las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="ndvi",
                        type=np.float32,
                        description="NDVI (vegetation index)",
                    )
                )
                las.ndvi = original_patch["ndvi"].astype(np.float32)
                added_dimensions.add("ndvi")
            except Exception as e:
                logger.warning(f"Could not add NDVI to LAZ: {e}")


def _add_return_number(las, original_patch: Dict, added_dimensions: set) -> None:
    """Add return number if available."""
    if "return_number" in original_patch and "return_number" not in added_dimensions:
        try:
            # Return number might already be a standard field
            if not hasattr(las, "return_number"):
                las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="return_number", type=np.uint8, description="Return number"
                    )
                )
                las.return_number = original_patch["return_number"].astype(np.uint8)
                added_dimensions.add("return_number")
        except Exception as e:
            logger.warning(f"Could not add return_number to LAZ: {e}")


def save_patch_multi_format(
    base_path: Path,
    data: Dict[str, np.ndarray],
    formats: List[str],
    original_patch: Optional[Dict] = None,
    lod_level: Optional[str] = None,
) -> int:
    """
    Save patch in multiple formats.

    Args:
        base_path: Base path without extension (e.g., "patch_0001")
        data: Patch data dictionary
        formats: List of format strings ('npz', 'hdf5', 'torch', 'laz')
        original_patch: Original patch for LAZ format (required if saving LAZ)
        lod_level: LOD level for NPZ format

    Returns:
        Number of files saved

    Raises:
        ImportError: If required libraries for requested formats are not installed
    """
    num_saved = 0

    for fmt in formats:
        fmt = fmt.strip().lower()

        if fmt == "npz":
            save_path = base_path.with_suffix(".npz")
            save_patch_npz(save_path, data, lod_level)
            num_saved += 1

        elif fmt == "hdf5":
            save_path = base_path.with_suffix(".h5")
            save_patch_hdf5(save_path, data)
            num_saved += 1

        elif fmt in ["pytorch", "torch"]:
            save_path = base_path.with_suffix(".pt")
            save_patch_torch(save_path, data)
            num_saved += 1

        elif fmt == "laz":
            if original_patch is None:
                logger.warning("Cannot save LAZ format: original_patch required")
                continue
            save_path = base_path.with_suffix(".laz")
            save_patch_laz(save_path, data, original_patch)
            num_saved += 1

        else:
            logger.warning(f"Unknown format '{fmt}', skipping")

    return num_saved


def save_enriched_tile_laz(
    save_path: Path,
    points: np.ndarray,
    classification: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    features: Dict[str, np.ndarray],
    original_las: Optional[laspy.LasData] = None,
    header: Optional[laspy.LasHeader] = None,
    input_rgb: Optional[np.ndarray] = None,
    input_nir: Optional[np.ndarray] = None,
) -> None:
    """
    Save a full enriched tile as LAZ with all computed features.

    This function preserves the original LAZ structure while adding computed
    features as extra dimensions. RGB colors come from orthophoto augmentation,
    while ASPRS classification codes are stored in the separate classification field.

    Args:
        save_path: Output LAZ file path
        points: (N, 3) XYZ coordinates in original projection
        classification: (N,) classification codes (potentially updated)
        intensity: (N,) intensity values [0-1]
        return_number: (N,) return numbers
        features: Dictionary of computed features to add as extra dimensions
        original_las: Original laspy.LasData object for header info (optional, for standard loading)
        header: Original header (optional, for chunked loading)
        input_rgb: RGB data if available (optional, for chunked loading)
        input_nir: NIR data if available (optional, for chunked loading)

    Raises:
        ImportError: If laspy is not installed
    """
    if not LASPY_AVAILABLE:
        raise ImportError(
            "laspy is required for LAZ format. " "Install with: pip install laspy"
        )

    # Determine appropriate point format based on available data
    has_rgb = (
        original_las is not None and hasattr(original_las, "red")
    ) or input_rgb is not None
    has_nir = (
        original_las is not None and hasattr(original_las, "nir")
    ) or input_nir is not None

    # Point format selection:
    # - Format 6: basic (no RGB, no NIR)
    # - Format 7: RGB but no NIR
    # - Format 8: RGB and NIR
    if has_rgb and has_nir:
        point_format = 8  # RGB + NIR
    elif has_rgb:
        point_format = 7  # RGB only
    else:
        point_format = 6  # Basic

    # Create new LAZ header
    if original_las is not None:
        # Use original LAS object's header
        new_header = laspy.LasHeader(
            version=original_las.header.version, point_format=point_format
        )
        new_header.offsets = original_las.header.offsets
        new_header.scales = original_las.header.scales
    elif header is not None:
        # Use provided header (from chunked loading)
        new_header = laspy.LasHeader(version=header.version, point_format=point_format)
        new_header.offsets = header.offsets
        new_header.scales = header.scales
    else:
        # Fallback: create a new header with default values
        new_header = laspy.LasHeader(version="1.4", point_format=point_format)
        new_header.offsets = [
            np.floor(points[:, 0].min()),
            np.floor(points[:, 1].min()),
            np.floor(points[:, 2].min()),
        ]
        new_header.scales = [0.001, 0.001, 0.001]

    # Create new LasData
    las = laspy.LasData(new_header)

    # Set coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Set standard fields
    las.intensity = (intensity * 65535.0).astype(np.uint16)
    las.return_number = return_number.astype(np.uint8)
    las.classification = classification.astype(np.uint8)

    # Set RGB colors from orthophoto augmentation
    # Classification is stored separately in the classification field (ASPRS codes)
    if original_las is not None and hasattr(original_las, "red"):
        # Copy RGB from original LAS object
        logger.debug("  Using RGB from original LAS file")
        las.red = original_las.red
        las.green = original_las.green
        las.blue = original_las.blue
    elif input_rgb is not None:
        # Use RGB from orthophotos/chunked loading
        logger.debug("  Using RGB from orthophoto augmentation")
        las.red = (input_rgb[:, 0] * 65535.0).astype(np.uint16)
        las.green = (input_rgb[:, 1] * 65535.0).astype(np.uint16)
        las.blue = (input_rgb[:, 2] * 65535.0).astype(np.uint16)  # Set NIR if available
    if original_las is not None and hasattr(original_las, "nir"):
        # Copy NIR from original LAS object
        las.nir = original_las.nir
    elif input_nir is not None:
        # Use NIR from chunked loading
        las.nir = (input_nir * 65535.0).astype(np.uint16)

    # Add computed features as extra dimensions
    # 🔧 FIX: Start with empty set - we want to OVERWRITE old features with newly computed ones
    # Don't initialize from original file to avoid skipping recomputed features
    added_dimensions = set()

    # Helper function to truncate feature names to 32 characters (LAS/LAZ limit)
    def truncate_name(name: str, max_len: int = 32) -> str:
        """Truncate feature name to fit LAS extra dimension name limit."""
        if len(name) <= max_len:
            return name
        # Try to abbreviate common long words
        abbreviations = {
            "eigenvalues": "eigval",
            "eigenvalue": "eigval",
            "neighborhood": "neigh",
            "likelihood": "like",
            "indicator": "ind",
            "roughness": "rough",
            "extent": "ext",
            "vertical": "vert",
            "height": "h",
            "above": "abv",
            "ground": "gnd",
            "points": "pts",
            "corner": "corn",
            "overhang": "ovhng",
            "surface": "surf",
        }
        truncated = name
        for long_word, short_word in abbreviations.items():
            truncated = truncated.replace(long_word, short_word)
        # If still too long, just cut it
        if len(truncated) > max_len:
            truncated = truncated[:max_len]
        return truncated

    # Log features being processed
    logger.debug(f"  Processing {len(features)} features for LAZ export")
    logger.debug(f"  Feature names: {list(features.keys())[:20]}...")  # First 20
    logger.debug(f"  Pre-existing extra dimensions: {sorted(added_dimensions)}")

    # Native dimensions of the chosen point format (x, y, z, intensity, red,
    # green, blue, nir, ...) — calling add_extra_dim() with a name that
    # collides with one of these raises "field '<name>' occurs more than
    # once" AND leaves the LasData's point record in a state where every
    # subsequent add_extra_dim() call also fails, regardless of name. A
    # `features` dict coming straight from FeatureOrchestrator commonly
    # includes 'red'/'green'/'blue'/'nir' alongside the geometric features,
    # so this must be checked before attempting to add anything, not just
    # caught after the fact.
    native_dimension_names = set(las.point_format.dimension_names)

    for feat_name, feat_data in features.items():
        if feat_name in ["points", "classification", "intensity", "return_number"]:
            logger.debug(f"  Skipping standard field: {feat_name}")
            continue  # Skip standard fields already set

        if feat_name in native_dimension_names:
            logger.debug(
                f"  Skipping '{feat_name}': already a native LAS dimension "
                f"for this point format"
            )
            continue

        if feat_name in added_dimensions:
            logger.debug(f"  Skipping duplicate: {feat_name}")
            continue  # Skip duplicates

        try:
            # Handle normals specially - they need to be split into 3 dimensions
            if (
                feat_name == "normals"
                and feat_data.ndim == 2
                and feat_data.shape[1] == 3
            ):
                for i, axis in enumerate(["x", "y", "z"]):
                    dim_name = f"normal_{axis}"
                    if dim_name not in added_dimensions:
                        las.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name=dim_name,
                                type=np.float32,
                                description=f"Normal vector {axis} component",
                            )
                        )
                        setattr(las, dim_name, feat_data[:, i].astype(np.float32))
                        added_dimensions.add(dim_name)
                continue

            # Skip multi-dimensional features that aren't normals
            if feat_data.ndim > 1:
                logger.debug(
                    f"Skipping multi-dimensional feature '{feat_name}' (shape: {feat_data.shape})"
                )
                continue

            # Truncate feature name if needed
            truncated_name = truncate_name(feat_name)
            if truncated_name != feat_name:
                logger.debug(
                    f"Truncated feature name: '{feat_name}' -> '{truncated_name}'"
                )

            # Skip if already added (after truncation)
            if truncated_name in added_dimensions:
                continue

            # Determine appropriate data type
            if feat_data.dtype in [np.float32, np.float64]:
                dtype = np.float32
            elif feat_data.dtype == np.int32:
                dtype = np.int32
            elif feat_data.dtype == np.uint8:
                dtype = np.uint8
            else:
                dtype = np.float32  # Default

            # Add extra dimension (description has 31-byte limit - needs null terminator)
            description = (
                feat_name[:31] if len(feat_name) <= 31 else truncated_name[:31]
            )
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=truncated_name, type=dtype, description=description
                )
            )
            setattr(las, truncated_name, feat_data.astype(dtype))
            added_dimensions.add(truncated_name)

        except Exception as e:
            logger.warning(f"Could not add feature '{feat_name}' to LAZ: {e}")

    # Write LAZ file
    las.write(str(save_path))
    logger.info(
        f"  ✓ Saved enriched tile: {save_path.name} ({len(points):,} points, {len(added_dimensions)} extra features)"
    )


def validate_format_support(formats: List[str]) -> Dict[str, bool]:
    """
    Check which formats are supported based on installed libraries.

    Args:
        formats: List of format strings to check

    Returns:
        Dictionary mapping format to boolean (supported or not)
    """
    support = {}

    for fmt in formats:
        fmt = fmt.strip().lower()

        if fmt == "npz":
            support[fmt] = True  # NumPy is always available
        elif fmt == "hdf5":
            support[fmt] = H5PY_AVAILABLE
        elif fmt in ["pytorch", "torch"]:
            support[fmt] = TORCH_AVAILABLE
        elif fmt == "laz":
            support[fmt] = LASPY_AVAILABLE
        else:
            support[fmt] = False

    return support
