"""
Infrared (NIR) Augmentation from IGN Orthophotos

This module provides functionality to augment LiDAR point clouds with Near-Infrared
(NIR) values from IGN's infrared orthophoto service.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from ..utils.normalization import normalize_nir

logger = logging.getLogger(__name__)

try:
    import requests
    from PIL import Image
    from io import BytesIO
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests or PIL not available - Infrared augmentation disabled")


class IGNInfraredFetcher:
    """
    Fetch infrared (NIR) values from IGN infrared orthophoto service for point clouds.
    
    Supports GPU-accelerated interpolation for significant speedup.
    """
    
    # IGN Géoplateforme WMS service for infrared orthophotos
    WMS_URL = "https://data.geopf.fr/wms-r"
    
    # Infrared layer (if available - may need to be adjusted based on IGN's service)
    # Note: IGN may have specific infrared layers - check their documentation
    LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.IRC"  # IRC = Infrared Color
    
    def __init__(self, cache_dir: Optional[Path] = None, use_gpu: bool = False):
        """
        Initialize infrared orthophoto fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded infrared orthophotos (disk)
            use_gpu: Enable GPU memory caching for faster access
                    (requires CuPy, provides significant speedup)
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "Infrared augmentation requires 'requests' and 'Pillow'. "
                "Install with: pip install requests Pillow"
            )
        
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU support
        self.use_gpu = use_gpu
        self.cp = None
        self.gpu_cache = None
        self.gpu_cache_order = []
        self.gpu_cache_max_size = 10  # Max tiles in GPU memory
        
        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_cache = {}
                logger.info("GPU tile caching enabled (Infrared augmentation)")
            except ImportError:
                logger.warning(
                    "GPU caching requested but CuPy unavailable. "
                    "Using CPU-only mode."
                )
                self.use_gpu = False
    
    def fetch_infrared_orthophoto(
        self,
        bbox: Tuple[float, float, float, float],
        width: int = 1024,
        height: int = 1024,
        crs: str = "EPSG:2154"
    ) -> Optional[np.ndarray]:
        """
        Fetch infrared orthophoto for a given bounding box.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in CRS coordinates
            width: Image width in pixels
            height: Image height in pixels
            crs: Coordinate reference system (default: EPSG:2154 Lambert 93)
        
        Returns:
            NIR image as numpy array [H, W] (single channel) or [H, W, 3] if IRC
            Returns None if failed
        """
        # Check cache first
        if self.cache_dir:
            cache_key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{width}x{height}"
            cache_file = self.cache_dir / f"nir_{cache_key}.png"
            
            if cache_file.exists():
                try:
                    img = Image.open(cache_file)
                    return np.array(img)
                except Exception as e:
                    logger.warning(f"Failed to load cached infrared orthophoto: {e}")
        
        # Build WMS GetMap request
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': self.LAYER,
            'CRS': crs,
            'BBOX': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
            'WIDTH': width,
            'HEIGHT': height,
            'FORMAT': 'image/png',
            'STYLES': ''
        }
        
        try:
            logger.debug(f"Fetching infrared orthophoto for bbox {bbox}")
            from ..utils.http_retry import get_with_retry
            response = get_with_retry(
                self.WMS_URL, params, timeout=30, operation_name="NIR orthophoto fetch",
            )
            
            # Load image
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            
            # Cache if enabled
            if self.cache_dir and cache_file:
                img.save(cache_file)
                logger.debug(f"Cached infrared orthophoto to {cache_file}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to fetch infrared orthophoto: {e}")
            return None
    
    def _get_cache_key(
        self, bbox: Tuple[float, float, float, float]
    ) -> str:
        """Generate cache key from bbox."""
        return f"{bbox[0]:.0f}_{bbox[1]:.0f}_{bbox[2]:.0f}_{bbox[3]:.0f}"
    
    def fetch_infrared_orthophoto_gpu(
        self,
        bbox: Tuple[float, float, float, float],
        width: int = 1024,
        height: int = 1024,
        crs: str = "EPSG:2154"
    ) -> 'cp.ndarray':
        """
        Fetch infrared tile and return as GPU array.
        
        Uses LRU cache in GPU memory for fast repeated access.
        Provides significant speedup over repeated CPU loading.
        
        Args:
            bbox: (xmin, ymin, xmax, ymax) in Lambert-93
            width: Image width in pixels
            height: Image height in pixels
            crs: Coordinate reference system
            
        Returns:
            nir_gpu: [H, W] or [H, W, 3] CuPy array (uint8)
        """
        if not self.use_gpu or self.cp is None:
            # Fallback to CPU
            nir_array = self.fetch_infrared_orthophoto(bbox, width, height, crs)
            if nir_array is None:
                return None
            return self.cp.asarray(nir_array)
        
        # Check GPU cache
        cache_key = self._get_cache_key(bbox)
        if cache_key in self.gpu_cache:
            # Cache hit - move to end (most recent)
            self.gpu_cache_order.remove(cache_key)
            self.gpu_cache_order.append(cache_key)
            logger.debug(f"GPU cache hit (infrared): {cache_key}")
            return self.gpu_cache[cache_key]
        
        # Cache miss - load from disk or download
        nir_array = self.fetch_infrared_orthophoto(bbox, width, height, crs)
        if nir_array is None:
            return None
            
        nir_gpu = self.cp.asarray(nir_array)
        
        # Add to GPU cache
        self.gpu_cache[cache_key] = nir_gpu
        self.gpu_cache_order.append(cache_key)
        
        # Evict oldest if cache full
        if len(self.gpu_cache) > self.gpu_cache_max_size:
            oldest_key = self.gpu_cache_order.pop(0)
            del self.gpu_cache[oldest_key]
            logger.debug(f"GPU cache evicted (infrared): {oldest_key}")
        
        cache_size = len(self.gpu_cache)
        logger.debug(f"GPU cache miss (infrared): {cache_key} (size: {cache_size})")
        return nir_gpu
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.use_gpu and self.gpu_cache is not None:
            self.gpu_cache.clear()
            self.gpu_cache_order.clear()
            logger.info("GPU cache cleared (infrared)")
    
    def augment_points_with_infrared(
        self,
        points: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        resolution: float = 0.2,
        default_value: int = 128
    ) -> np.ndarray:
        """
        Augment point cloud with infrared values from orthophoto.
        
        Args:
            points: Point cloud array [N, 3] (X, Y, Z)
            bbox: Bounding box (xmin, ymin, xmax, ymax). If None, computed from points
            resolution: Orthophoto resolution in meters per pixel (default: 0.2m = 20cm)
            default_value: Default infrared value if fetch fails (0-255)
        
        Returns:
            Infrared values array [N] with values in range [0, 255]
        """
        # Compute bounding box if not provided
        if bbox is None:
            bbox = (
                points[:, 0].min(),
                points[:, 1].min(),
                points[:, 0].max(),
                points[:, 1].max()
            )
        
        # Calculate image dimensions based on bbox and resolution
        width = int((bbox[2] - bbox[0]) / resolution)
        height = int((bbox[3] - bbox[1]) / resolution)
        
        # Limit size for memory efficiency
        max_dim = 2048
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Fetch infrared orthophoto
        nir_img = self.fetch_infrared_orthophoto(bbox, width, height)
        
        if nir_img is None:
            logger.warning(
                f"Using default infrared value {default_value} "
                f"(infrared orthophoto fetch failed)"
            )
            return np.full(len(points), default_value, dtype=np.uint8)
        
        # Handle multi-channel IRC images (extract NIR channel)
        if len(nir_img.shape) == 3:
            # IRC typically has NIR in red channel
            # Check if it's IRC (Infrared Color) format
            nir_img = nir_img[:, :, 0]  # Extract first channel (typically NIR)
            logger.debug("Extracted NIR channel from IRC image")
        
        # Map points to image coordinates
        # Note: Image origin is top-left, so we need to flip Y
        x_norm = (points[:, 0] - bbox[0]) / (bbox[2] - bbox[0])
        y_norm = (points[:, 1] - bbox[1]) / (bbox[3] - bbox[1])
        
        # Flip Y for image coordinates (top-left origin)
        y_norm = 1.0 - y_norm
        
        # Convert to pixel coordinates
        px = (x_norm * (width - 1)).astype(int)
        py = (y_norm * (height - 1)).astype(int)
        
        # Clamp to image bounds
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        
        # Sample infrared values from orthophoto
        nir_values = nir_img[py, px]
        
        return nir_values.astype(np.uint8)
    
    def fetch_for_points(
        self,
        points: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        resolution: float = 0.2
    ) -> np.ndarray:
        """
        Fetch infrared values for points (alias for augment_points_with_infrared).
        
        This method provides compatibility with the orchestrator interface.
        
        Args:
            points: Point cloud array [N, 3] (X, Y, Z)
            bbox: Bounding box (xmin, ymin, xmax, ymax). If None, computed from points
            resolution: Orthophoto resolution in meters per pixel (default: 0.2m = 20cm)
        
        Returns:
            Infrared values array [N] with values in range [0, 255]
        """
        return self.augment_points_with_infrared(
            points=points,
            bbox=bbox,
            resolution=resolution
        )


def add_infrared_to_patch(
    patch: dict,
    nir_fetcher: IGNInfraredFetcher,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> dict:
    """
    Add infrared values to a patch dictionary.
    
    Args:
        patch: Patch dictionary with 'points' key
        nir_fetcher: Infrared fetcher instance
        bbox: Optional bounding box for the tile
    
    Returns:
        Updated patch dictionary with 'nir' key added
    """
    # Get absolute point coordinates if centered
    # (patches are usually centered around origin)
    points_abs = patch['points'].copy()
    
    # If bbox provided, restore absolute coordinates
    if bbox is not None:
        center = np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
            0
        ])
        points_abs[:, :2] += center[:2]
    
    # Fetch infrared values
    nir = nir_fetcher.augment_points_with_infrared(points_abs, bbox=bbox)
    
    # Normalize to [0, 1] for consistency with other features
    patch['nir'] = normalize_nir(nir, use_gpu=False)
    
    return patch


# Example usage functions

def augment_tile_with_infrared(
    laz_file: Path,
    output_dir: Path,
    cache_dir: Optional[Path] = None
):
    """
    Process a LAZ file and save with infrared augmentation.
    
    Args:
        laz_file: Path to input LAZ file
        output_dir: Output directory for augmented file
        cache_dir: Cache directory for infrared orthophotos
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("Infrared augmentation requires laspy. Install with: pip install laspy")
    
    # Load LAZ
    las = laspy.read(str(laz_file))
    points = np.vstack([las.x, las.y, las.z]).T
    
    # Compute bbox
    bbox = (
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max()
    )
    
    # Fetch infrared values
    fetcher = IGNInfraredFetcher(cache_dir=cache_dir)
    nir = fetcher.augment_points_with_infrared(points, bbox=bbox)
    
    logger.info(f"Augmented {len(points)} points with NIR from IGN infrared orthophoto")
    
    # Add infrared to LAZ as extra dimension
    try:
        from laspy import ExtraBytesParams
        las.add_extra_dim(ExtraBytesParams(name='nir', type=np.uint8))
        las.nir = nir
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / laz_file.name
        las.write(output_path)
        logger.info(f"Saved infrared-augmented LAZ to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save infrared-augmented LAZ: {e}")
        raise
