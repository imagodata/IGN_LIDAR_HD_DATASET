"""
PTv3 / Pointcept formatter for IGN LiDAR HD aerial patches.

Produces a dataset layout compatible with Pointcept's `DefaultDataset` so
patches can be consumed directly by Point Transformer V3 (PTv3) and other
Pointcept-family models without an extra conversion step.

Layout (per patch):
    <output_root>/
        meta.json
        <split>/<patch_id>/
            coord.npy     # (N, 3) float32 — XY centered on patch, Z anchored at z_min
            feat.npy      # (N, F) float32 — feature stack chosen via `feat_keys`
            segment.npy   # (N,)   int16   — contiguous Lidar HD labels (or -1 for ignore)
            offset.npy    # (3,)   float64 — global offset (extractor + formatter);
                          #                  coord_absolute = coord + offset (Lambert93).
                          #                  Required to re-project predictions onto the
                          #                  source tile at inference (overlap + resampling
                          #                  make an exact point index impossible).
            metadata.json # CRS, source bbox, absolute patch bbox and coordinate contract.
            source_indices.npy  # optional indices into the source cloud.

The formatter does NOT voxelize. PTv3 voxelizes at training time via Pointcept's
`GridSample` transform (recommended `grid_size=0.15` for aerial Lidar HD).

Reference:
    - Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024.
    - https://github.com/Pointcept/Pointcept
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...classification_schema import (
    LIDARHD_7CL_IGNORE_INDEX,
    LIDARHD_7CL_NAMES,
    LIDARHD_7CL_TO_ASPRS,
    remap_asprs_to_lidarhd_7cl,
)
from .base_formatter import BaseFormatter

logger = logging.getLogger(__name__)


# Feature keys supported as entries in `feat_keys`. The formatter looks them up
# in the patch dict (or expands grouped keys like "normals").
SUPPORTED_FEAT_KEYS: Dict[str, int] = {
    "intensity": 1,
    "return_number": 1,
    "num_returns": 1,
    "normals": 3,
    "ndvi": 1,
    "nir": 1,
    "rgb": 3,
    "height_above_ground": 1,
    # Geometric descriptors (LOD2/LOD3 building discrimination). Already bounded
    # ~[0,1] by the feature engine, so _scale_feat passes them through.
    "curvature": 1,
    "horizontality": 1,
    "planarity": 1,
    "linearity": 1,
    "verticality": 1,
    "wall_score": 1,
    "roof_score": 1,
}

LABEL_SCHEMAS = ("lidar_hd_7cl_contiguous", "raw")
LAYOUTS = ("folder", "pth")
# Intensity normalisation. "linear" is the historical behaviour and stays the
# default: the 9D/12D datasets already used to train shipped models were built
# with it, and silently switching would break their reproducibility. "log"
# (opt-in, used by the 17D preset) compresses the long uint16 tail so the bulk
# of Lidar HD returns — clustered in the low thousands — actually spreads over
# the usable range instead of being crushed near 0.
INTENSITY_SCALINGS = ("linear", "log")


class Ptv3Formatter(BaseFormatter):
    """Format patches for PTv3 / Pointcept training (aerial LiDAR HD).

    Notes:
        - `num_points` is intentionally ignored (PTv3 voxelizes at train time).
        - XY is centered on the patch centroid; Z is shifted so z_min = 0 to
          give PTv3 a stable height anchor without exposing absolute elevations.
    """

    def __init__(
        self,
        feat_keys: Sequence[str] = ("intensity", "return_number", "normals"),
        label_schema: str = "lidar_hd_7cl_contiguous",
        layout: str = "folder",
        center_xy: bool = True,
        anchor_z_min: bool = True,
        ignore_index: int = LIDARHD_7CL_IGNORE_INDEX,
        intensity_scaling: str = "linear",
    ):
        super().__init__(num_points=-1, normalize=False, standardize_features=False)

        if label_schema not in LABEL_SCHEMAS:
            raise ValueError(
                f"label_schema must be one of {LABEL_SCHEMAS}, got {label_schema!r}"
            )
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if intensity_scaling not in INTENSITY_SCALINGS:
            raise ValueError(
                f"intensity_scaling must be one of {INTENSITY_SCALINGS}, "
                f"got {intensity_scaling!r}"
            )
        for key in feat_keys:
            if key not in SUPPORTED_FEAT_KEYS:
                raise ValueError(
                    f"Unsupported feat key {key!r}. Supported: {sorted(SUPPORTED_FEAT_KEYS)}"
                )

        self.feat_keys = tuple(feat_keys)
        self.label_schema = label_schema
        self.layout = layout
        self.center_xy = center_xy
        self.anchor_z_min = anchor_z_min
        self.ignore_index = int(ignore_index)
        self.intensity_scaling = intensity_scaling

    # ------------------------------------------------------------------ format

    def format_patch(self, patch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Return a Pointcept-ready record for a single patch.

        Args:
            patch: Dict from the upstream pipeline. Must contain `points` (N,3)
                and `labels` (N,). Optional: `normals`, `intensity`,
                `return_number`, `num_returns`, `curvature`, `horizontality`,
                `ndvi`, `nir`, `rgb`, `height_above_ground`. Identifiers expected:
                `patch_id` and/or `tile_id`.

        Returns:
            Dict with keys: coord (N,3 float32), feat (N,F float32),
            segment (N, int16), name (str), tile_id (Optional[str]),
            feat_layout (List[str]).
        """
        if "points" not in patch:
            raise KeyError("patch missing required 'points' key")
        if "labels" not in patch:
            raise KeyError("patch missing required 'labels' key")

        coord, formatter_offset = self._build_coord(patch["points"])
        patch_center = self._get_patch_center(patch)
        offset = patch_center + formatter_offset
        absolute_coord = coord.astype(np.float64) + offset
        patch_bbox = self._bbox_3d(absolute_coord)
        source_bbox = self._get_source_bbox(patch, patch_bbox)
        source_indices = self._get_source_indices(patch, coord.shape[0])
        crs = patch.get("_crs", patch.get("crs", "EPSG:2154"))
        feat, feat_layout = self._build_feat(patch, coord.shape[0])
        segment = self._build_segment(patch["labels"])

        return {
            "coord": coord,
            "feat": feat,
            "segment": segment,
            "offset": offset,
            "patch_bbox": patch_bbox,
            "source_bbox": source_bbox,
            "crs": None if crs is None else str(crs),
            "source_indices": source_indices,
            "name": str(patch.get("patch_id", patch.get("name", "patch"))),
            "tile_id": patch.get("tile_id"),
            "feat_layout": feat_layout,
        }

    # --------------------------------------------------------------- writers

    def write_patch(self, formatted: Dict[str, Any], output_dir: Path) -> Path:
        """Persist a single formatted record to disk.

        For `layout="folder"`, writes `<output_dir>/<name>/{coord,feat,segment}.npy`.
        For `layout="pth"`, writes `<output_dir>/<name>.pth` (torch.save).
        Returns the path written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = formatted["name"]

        if self.layout == "folder":
            tile_dir = output_dir / name
            tile_dir.mkdir(parents=True, exist_ok=True)
            np.save(tile_dir / "coord.npy", formatted["coord"].astype(np.float32))
            np.save(tile_dir / "feat.npy", formatted["feat"].astype(np.float32))
            np.save(tile_dir / "segment.npy", formatted["segment"].astype(np.int16))
            # offset = [cx, cy, zmin] : coord_absolute = coord + offset. Additif,
            # ignoré par Pointcept (charge coord/feat/segment) ; sert au writeback
            # des prédictions vers la tuile source à l'inférence.
            if formatted.get("offset") is not None:
                np.save(tile_dir / "offset.npy", np.asarray(formatted["offset"], dtype=np.float64))
            metadata = self._coordinate_metadata(formatted)
            (tile_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True)
            )
            if formatted.get("source_indices") is not None:
                np.save(
                    tile_dir / "source_indices.npy",
                    np.asarray(formatted["source_indices"], dtype=np.int64),
                )
            return tile_dir

        # pth layout
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "layout='pth' requires PyTorch. Install torch or use layout='folder'."
            ) from exc
        target = output_dir / f"{name}.pth"
        record = {
            "coord": torch.from_numpy(formatted["coord"].astype(np.float32)),
            "feat": torch.from_numpy(formatted["feat"].astype(np.float32)),
            "segment": torch.from_numpy(formatted["segment"].astype(np.int64)),
            "name": name,
        }
        if formatted.get("offset") is not None:
            record["offset"] = torch.from_numpy(np.asarray(formatted["offset"], dtype=np.float64))
        record["metadata"] = self._coordinate_metadata(formatted)
        if formatted.get("source_indices") is not None:
            record["source_indices"] = torch.from_numpy(
                np.asarray(formatted["source_indices"], dtype=np.int64)
            )
        torch.save(record, target)
        return target

    def write_meta(self, output_root: Path, num_patches_per_split: Optional[Dict[str, int]] = None) -> Path:
        """Write `meta.json` at the dataset root. Idempotent.

        Records the label schema, feature layout and split sizes so the
        Pointcept dataset class can sanity-check the data it loads.
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        if self.label_schema == "lidar_hd_7cl_contiguous":
            label_names = list(LIDARHD_7CL_NAMES)
            inverse = {str(k): v for k, v in LIDARHD_7CL_TO_ASPRS.items()}
        else:
            label_names = []
            inverse = {}

        meta = {
            "schema_version": "1.1",
            "formatter": "Ptv3Formatter",
            "label_schema": self.label_schema,
            "label_names": label_names,
            "label_to_asprs": inverse,
            "ignore_index": self.ignore_index,
            "feat_keys": list(self.feat_keys),
            "feat_dim": sum(SUPPORTED_FEAT_KEYS[k] for k in self.feat_keys),
            "intensity_scaling": self.intensity_scaling,
            "layout": self.layout,
            "coord_dim": 3,
            "center_xy": self.center_xy,
            "anchor_z_min": self.anchor_z_min,
            "coordinate_contract": "coord_absolute = coord + offset",
            "patch_metadata": "metadata.json",
            "default_crs": "EPSG:2154",
            "splits": num_patches_per_split or {},
        }
        target = output_root / "meta.json"
        target.write_text(json.dumps(meta, indent=2, sort_keys=True))
        return target

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _get_patch_center(patch: Dict[str, Any]) -> np.ndarray:
        """Return the offset already subtracted by the patch extractor."""
        center = np.asarray(
            patch.get("_patch_center", np.zeros(3)), dtype=np.float64
        ).ravel()
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("_patch_center must contain three finite values")
        return center

    @staticmethod
    def _bbox_3d(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            raise ValueError("cannot compute a bounding box for an empty patch")
        return np.concatenate((points.min(axis=0), points.max(axis=0))).astype(
            np.float64
        )

    @staticmethod
    def _get_source_bbox(
        patch: Dict[str, Any], patch_bbox: np.ndarray
    ) -> np.ndarray:
        source_bbox = patch.get("_source_bbox", patch.get("source_bbox"))
        if source_bbox is None:
            return patch_bbox.copy()
        source_bbox = np.asarray(source_bbox, dtype=np.float64).ravel()
        if source_bbox.shape != (6,) or not np.isfinite(source_bbox).all():
            raise ValueError(
                "_source_bbox/source_bbox must be six finite values "
                "(xmin, ymin, zmin, xmax, ymax, zmax)"
            )
        return source_bbox

    @staticmethod
    def _get_source_indices(
        patch: Dict[str, Any], n_points: int
    ) -> Optional[np.ndarray]:
        source_indices = patch.get("source_indices")
        if source_indices is None:
            return None
        source_indices = np.asarray(source_indices)
        if source_indices.ndim != 1 or len(source_indices) != n_points:
            raise ValueError(
                "source_indices must be a 1-D array with one entry per point"
            )
        if not np.issubdtype(source_indices.dtype, np.integer):
            raise ValueError("source_indices must contain integers")
        return source_indices.astype(np.int64, copy=False)

    @staticmethod
    def _coordinate_metadata(formatted: Dict[str, Any]) -> Dict[str, Any]:
        """Build the JSON-safe, versioned coordinate contract for one patch."""
        return {
            "schema_version": "1.0",
            "coordinate_contract": "coord_absolute = coord + offset",
            "crs": formatted.get("crs"),
            "source_bbox": np.asarray(
                formatted["source_bbox"], dtype=np.float64
            ).tolist(),
            "patch_bbox": np.asarray(
                formatted["patch_bbox"], dtype=np.float64
            ).tolist(),
            "tile_id": formatted.get("tile_id"),
            "has_source_indices": formatted.get("source_indices") is not None,
        }

    def _build_coord(self, points: np.ndarray) -> tuple:
        """Recenter coords and return ``(coord_f32, offset_f64)``.

        The returned offset only represents this formatter's transform. The
        caller adds the extractor's ``_patch_center`` before persisting it, so
        the public ``offset`` is global and ``coord + offset`` is absolute.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N,3), got {points.shape}")
        # Center in float64 — Lambert93 coords are ~7e6 so float32 precision (~0.5m
        # at that magnitude) corrupts the centroid before we can subtract it.
        coord = points.astype(np.float64, copy=True)
        offset = np.zeros(3, dtype=np.float64)
        if self.center_xy:
            offset[:2] = coord[:, :2].mean(axis=0)
            coord[:, :2] -= offset[:2]
        if self.anchor_z_min:
            offset[2] = coord[:, 2].min()
            coord[:, 2] -= offset[2]
        return coord.astype(np.float32), offset

    def _build_feat(self, patch: Dict[str, np.ndarray], n_points: int) -> tuple:
        cols: List[np.ndarray] = []
        layout: List[str] = []
        for key in self.feat_keys:
            arr = patch.get(key)
            width = SUPPORTED_FEAT_KEYS[key]
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                # Missing optional feature: zero-fill so downstream tensor shape stays stable.
                cols.append(np.zeros((n_points, width), dtype=np.float32))
                layout.extend(self._expand_layout(key, width))
                logger.debug("ptv3_formatter: feat '%s' missing — zero-filling %d col(s)", key, width)
                continue

            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if arr.shape[0] != n_points:
                raise ValueError(
                    f"feature {key!r} length {arr.shape[0]} != n_points {n_points}"
                )
            if arr.shape[1] != width:
                raise ValueError(
                    f"feature {key!r} width {arr.shape[1]} != expected {width}"
                )

            cols.append(self._scale_feat(key, arr.astype(np.float32)))
            layout.extend(self._expand_layout(key, width))

        if not cols:
            # Edge case: empty feat_keys → minimal 1-D zero feat so PTv3 still receives a tensor.
            return np.zeros((n_points, 1), dtype=np.float32), ["zero"]
        return np.concatenate(cols, axis=1).astype(np.float32), layout

    @staticmethod
    def _expand_layout(key: str, width: int) -> List[str]:
        if width == 1:
            return [key]
        if key == "normals":
            return ["normal_x", "normal_y", "normal_z"]
        if key == "rgb":
            return ["r", "g", "b"]
        return [f"{key}_{i}" for i in range(width)]

    def _scale_feat(self, key: str, arr: np.ndarray) -> np.ndarray:
        # Per-feature scaling kept simple and deterministic. PTv3 prefers
        # bounded, ~unit-scale inputs but is robust to small deviations.
        if key == "intensity":
            # Common Lidar HD range is uint16 [0, ~65535]; normalize to [0, 1].
            if self.intensity_scaling == "log":
                return np.log1p(np.clip(arr, 0.0, 65535.0)) / np.log1p(65535.0)
            return np.clip(arr / 65535.0, 0.0, 1.0)
        if key in ("return_number", "num_returns"):
            return np.clip(arr / 7.0, 0.0, 1.0)  # spec caps at 7 returns
        if key in ("rgb", "nir"):
            # FeatureOrchestrator._add_rgb_features()/_add_nir_features() already
            # normalize fetched BD ORTHO/IRC values to float32 [0, 1] via
            # normalize_rgb()/normalize_nir() before they reach the patch dict —
            # dividing by 255 here again crushed rgb/nir into ~[0, 0.004],
            # destroying essentially all color/spectral signal. Just clip
            # defensively, same as the other pre-scaled geometric descriptors.
            return np.clip(arr, 0.0, 1.0)
        if key == "ndvi":
            # NDVI ∈ [-1, 1] → remap to [0, 1] so spectral channels share scale.
            return np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
        if key == "height_above_ground":
            # AGL > 100 m is rare for aerial Lidar HD (tall trees ≤ 60 m,
            # buildings ≤ 80 m). Clip + /100 gives a bounded, deterministic input
            # without per-patch statistics (which collapse on flat ground patches).
            return np.clip(arr / 100.0, 0.0, 1.0)
        if key in (
            "curvature",
            "horizontality",
            "planarity",
            "linearity",
            "verticality",
            "wall_score",
            "roof_score",
        ):
            # Geometric descriptors are already ~[0,1]; clip defensively.
            return np.clip(arr, 0.0, 1.0)
        # normals already unit vectors.
        return arr

    def _build_segment(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            labels = labels.ravel()
        if self.label_schema == "lidar_hd_7cl_contiguous":
            return remap_asprs_to_lidarhd_7cl(labels)
        return labels.astype(np.int16)
