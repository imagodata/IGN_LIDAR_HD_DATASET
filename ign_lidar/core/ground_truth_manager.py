"""
Ground Truth Data Management Module

This module handles fetching, caching, and managing ground truth data from
external sources (WFS, BD TOPO, etc.) for LiDAR processing.

Extracted from LiDARProcessor to reduce complexity and improve maintainability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import laspy
import numpy as np

logger = logging.getLogger(__name__)


class GroundTruthManager:
    """
    Manages ground truth data fetching and caching for LiDAR processing.
    
    Responsibilities:
    - Pre-fetch ground truth data for tiles
    - Cache ground truth data for reuse
    - Estimate bounding boxes from LAZ headers
    - Coordinate with WFS/BD TOPO services
    
    This class extracts ground truth management from LiDARProcessor to
    improve separation of concerns and testability.
    """

    # Max number of tiles kept in the ground truth cache. Only the current and
    # the prefetched next tile are ever read back, so a small bound is enough.
    MAX_CACHED_TILES = 4

    def __init__(
        self,
        data_sources_config: Optional[Dict] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize ground truth manager.
        
        Args:
            data_sources_config: Configuration for data sources (bd_topo, wfs, etc.)
            cache_dir: Directory for caching ground truth data
        """
        self.data_sources_config = data_sources_config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._ground_truth_cache = {}
        
        logger.debug(
            f"GroundTruthManager initialized with sources: "
            f"{list(self.data_sources_config.keys())}"
        )
    
    def prefetch_ground_truth_for_tile(self, laz_file: Path) -> Optional[dict]:
        """
        Pre-fetch ground truth data for a single tile.
        
        Args:
            laz_file: Path to LAZ file to prefetch data for
            
        Returns:
            Dictionary containing fetched ground truth data, or None if failed
        """
        try:
            # Quick bbox estimation from LAZ header (fast read)
            las = laspy.read(str(laz_file))
            
            # Get bounding box from header
            bbox = {
                'minx': float(las.header.x_min),
                'maxx': float(las.header.x_max),
                'miny': float(las.header.y_min),
                'maxy': float(las.header.y_max)
            }
            
            # Check if we need to fetch ground truth
            if not self.data_sources_config:
                return None
                
            # Import here to avoid circular imports
            from ..io.wfs_ground_truth import WFSGroundTruthFetcher
            
            # Create fetcher with config
            fetcher = WFSGroundTruthFetcher(
                config=self.data_sources_config,
                cache_dir=self.cache_dir
            )
            
            # Fetch data for bounding box
            ground_truth_data = fetcher.fetch_for_bbox(bbox)
            
            # Cache it
            # ⚠️ Bounded: only the in-flight tiles (current + prefetched next)
            # are ever looked up, but this dict used to keep the ground truth
            # of every tile of the run alive. A miss is harmless (the caller
            # gets None and refetches).
            tile_key = laz_file.stem
            self._ground_truth_cache[tile_key] = ground_truth_data
            while len(self._ground_truth_cache) > self.MAX_CACHED_TILES:
                self._ground_truth_cache.pop(next(iter(self._ground_truth_cache)))

            logger.debug(
                f"Prefetched ground truth for {laz_file.name}: "
                f"{len(ground_truth_data.get('buildings', []))} buildings"
            )
            
            return ground_truth_data
            
        except Exception as e:
            logger.warning(
                f"Failed to prefetch ground truth for {laz_file.name}: {e}"
            )
            return None
    
    def prefetch_ground_truth_batch(
        self,
        laz_files: List[Path],
        show_progress: bool = True
    ) -> Dict[str, dict]:
        """
        Pre-fetch ground truth data for multiple tiles.
        
        Args:
            laz_files: List of LAZ files to prefetch data for
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping tile names to ground truth data
        """
        results = {}
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(laz_files, desc="Prefetching ground truth")
        else:
            iterator = laz_files
        
        for laz_file in iterator:
            ground_truth = self.prefetch_ground_truth_for_tile(laz_file)
            if ground_truth:
                results[laz_file.stem] = ground_truth
        
        logger.info(
            f"Prefetched ground truth for {len(results)}/{len(laz_files)} tiles"
        )
        
        return results
    
    def get_cached_ground_truth(self, tile_key: str) -> Optional[dict]:
        """
        Get cached ground truth data for a tile.
        
        Args:
            tile_key: Tile identifier (usually stem of filename)
            
        Returns:
            Cached ground truth data, or None if not available
        """
        return self._ground_truth_cache.get(tile_key)
    
    def clear_cache(self):
        """Clear the ground truth cache."""
        self._ground_truth_cache.clear()
        logger.debug("Ground truth cache cleared")
    
    def estimate_bbox_from_laz_header(self, laz_file: Path) -> Dict[str, float]:
        """
        Quickly estimate bounding box from LAZ header without reading points.
        
        Args:
            laz_file: Path to LAZ file
            
        Returns:
            Dictionary with minx, maxx, miny, maxy keys
        """
        try:
            las = laspy.read(str(laz_file))
            return {
                'minx': float(las.header.x_min),
                'maxx': float(las.header.x_max),
                'miny': float(las.header.y_min),
                'maxy': float(las.header.y_max)
            }
        except Exception as e:
            logger.error(f"Failed to read LAZ header from {laz_file.name}: {e}")
            return {}
