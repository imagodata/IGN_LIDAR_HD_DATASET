"""
IGN LiDAR HD Batch Download Module

This module provides functionality to download IGN LiDAR HD tiles
based on geographic coordinates, administrative boundaries, or tile lists.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import requests
from urllib.parse import urljoin
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .io.metadata import MetadataManager
from .datasets.strategic_locations import STRATEGIC_LOCATIONS

logger = logging.getLogger(__name__)


class IGNLiDARDownloader:
    """
    IGN LiDAR HD batch downloader with position-based tile discovery.
    """
    
    # IGN LiDAR HD WFS service URL for tile metadata
    # Note: namespace renommé en 2025 → IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle
    # Le serveur lit BBOX en ordre lon,lat,lon,lat avec EPSG:4326.
    WFS_URL = ("https://data.geopf.fr/wfs/wfs?SERVICE=WFS&REQUEST=GetFeature&"
               "VERSION=2.0.0&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle&"
               "OUTPUTFORMAT=application/json&SRSNAME=EPSG:4326")
    
    # IGN LiDAR HD download base URL  
    DOWNLOAD_BASE_URL = "https://data.geopf.fr/telechargement/download/"
    
    def __init__(self, output_dir: Path, max_concurrent: int = 3):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded tiles
            max_concurrent: Maximum concurrent downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self._tile_cache: Optional[Dict] = None
        self.metadata_manager = MetadataManager(self.output_dir)
        self._tile_location_map: Dict[str, Dict] = {}  # Map tile names to locations
        
    def fetch_available_tiles(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Fetch available LiDAR HD tiles from IGN WFS service.
        
        Args:
            bbox: Optional bounding box (xmin, ymin, xmax, ymax) in WGS84
            
        Returns:
            GeoJSON response with available tiles
        """
        url = self.WFS_URL
        
        # Add bounding box filter if provided
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            bbox_param = f"&BBOX={xmin},{ymin},{xmax},{ymax},EPSG:4326"
            url += bbox_param
            
        try:
            logger.info(f"Fetching tile information from WFS service...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved {len(data.get('features', []))} tiles from WFS")
            
            # Cache the result
            self._tile_cache = data
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch tiles from WFS: {e}")
            return {"type": "FeatureCollection", "features": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WFS response: {e}")
            return {"type": "FeatureCollection", "features": []}
    
    def get_tile_download_url(self, tile_name: str) -> Optional[str]:
        """
        Get download URL for a specific tile from WFS metadata.
        
        Args:
            tile_name: Name of the tile (e.g., "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69")
            
        Returns:
            Download URL if available
        """
        if not self._tile_cache:
            self.fetch_available_tiles()
            
        if not self._tile_cache:
            return None
            
        # Clean tile name for comparison (remove extensions)
        clean_tile_name = tile_name.replace('.laz', '').replace('.LAZ', '')
        
        # Search for tile in cached data
        for feature in self._tile_cache.get('features', []):
            properties = feature.get('properties', {})
            
            # Check the 'name' field which contains the tile filename
            if 'name' in properties:
                tile_filename = properties['name']
                # Remove extension for comparison
                clean_filename = tile_filename.replace('.copc.laz', '').replace('.laz', '')
                
                # Match by tile name (flexible matching)
                if (clean_tile_name in clean_filename or 
                    clean_filename in clean_tile_name or
                    self._tiles_match(clean_tile_name, clean_filename)):
                    
                    # Return the direct download URL
                    if 'url' in properties:
                        return properties['url']
        
        return None
    
    def _tiles_match(self, name1: str, name2: str) -> bool:
        """Check if two tile names refer to the same tile."""
        # Extract coordinate patterns
        import re
        
        # Try to extract X and Y coordinates from both names
        pattern = r'(\d{4})_(\d{4})'
        
        match1 = re.search(pattern, name1)
        match2 = re.search(pattern, name2)
        
        if match1 and match2:
            return match1.groups() == match2.groups()
        
        return False
        
    def lambert93_to_tile_coords(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert Lambert 93 coordinates to tile grid coordinates.
        
        Args:
            x: X coordinate in Lambert 93
            y: Y coordinate in Lambert 93
            
        Returns:
            Tuple of (tile_x, tile_y) in hectometers
        """
        # IGN tiles are 1km x 1km, coordinates in hectometers
        tile_x = int(x // 1000) * 10
        tile_y = int(y // 1000) * 10
        return tile_x, tile_y
    
    def tile_coords_to_filename(self, tile_x: int, tile_y: int) -> str:
        """
        Generate IGN LiDAR HD filename from tile coordinates.
        
        Args:
            tile_x: Tile X coordinate (hectometers)
            tile_y: Tile Y coordinate (hectometers)
            
        Returns:
            Standard IGN LiDAR HD filename
        """
        return f"HD_LIDARHD_FXX_{tile_x:04d}_{tile_y:04d}_PTS_C_LAMB93_IGN69.laz"
    
    def get_tiles_in_bbox(self, xmin: float, ymin: float, 
                         xmax: float, ymax: float) -> List[Tuple[int, int]]:
        """
        Get all tile coordinates within a bounding box.
        
        Args:
            xmin, ymin, xmax, ymax: Bounding box in Lambert 93
            
        Returns:
            List of (tile_x, tile_y) coordinates
        """
        tiles = []
        
        # Convert bbox to tile grid
        min_tile_x, min_tile_y = self.lambert93_to_tile_coords(xmin, ymin)
        max_tile_x, max_tile_y = self.lambert93_to_tile_coords(xmax, ymax)
        
        # Generate all tiles in the grid
        for tile_x in range(min_tile_x, max_tile_x + 10, 10):
            for tile_y in range(min_tile_y, max_tile_y + 10, 10):
                tiles.append((tile_x, tile_y))
        
        return tiles
    
    def check_tile_availability(self, tile_x: int, tile_y: int) -> bool:
        """
        Check if a tile is available for download.
        
        Args:
            tile_x: Tile X coordinate (hectometers)
            tile_y: Tile Y coordinate (hectometers)
            
        Returns:
            True if tile is available
        """
        # This is a simplified check - in practice, you'd query IGN's WFS service
        # For now, we'll simulate availability based on known coverage
        
        # Approximate coverage of IGN LiDAR HD (simplified)
        # Most of metropolitan France is covered
        if (100 <= tile_x <= 1300) and (6000 <= tile_y <= 7200):
            # Exclude some water bodies and border areas
            if tile_x < 200 and tile_y > 7000:  # English Channel
                return False
            if tile_x > 1200 and tile_y < 6200:  # Mediterranean 
                return False
            return True
        
        return False
    
    def find_tiles_by_position(self, x: float, y: float, 
                              radius_km: float = 5.0) -> List[Dict]:
        """
        Find available tiles around a given position.
        
        Args:
            x: X coordinate in Lambert 93
            y: Y coordinate in Lambert 93
            radius_km: Search radius in kilometers
            
        Returns:
            List of tile information dictionaries
        """
        # Create bounding box around position
        radius_m = radius_km * 1000
        xmin, ymin = x - radius_m, y - radius_m
        xmax, ymax = x + radius_m, y + radius_m
        
        # Get tiles in area
        tile_coords = self.get_tiles_in_bbox(xmin, ymin, xmax, ymax)
        
        # Check availability and build result
        available_tiles = []
        for tile_x, tile_y in tile_coords:
            if self.check_tile_availability(tile_x, tile_y):
                filename = self.tile_coords_to_filename(tile_x, tile_y)
                
                # Calculate tile center coordinates
                center_x = tile_x * 100 + 500  # Convert to meters + center offset
                center_y = tile_y * 100 + 500
                
                # Calculate distance from query point
                distance_km = ((center_x - x)**2 + (center_y - y)**2)**0.5 / 1000
                
                tile_info = {
                    'filename': filename,
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'center_x': center_x,
                    'center_y': center_y,
                    'distance_km': round(distance_km, 2),
                    'bbox': [
                        tile_x * 100, tile_y * 100,
                        tile_x * 100 + 1000, tile_y * 100 + 1000
                    ]
                }
                available_tiles.append(tile_info)
        
        # Sort by distance
        available_tiles.sort(key=lambda t: t['distance_km'])
        
        return available_tiles
    
    def _get_tile_location_info(self, filename: str) -> Optional[Dict]:
        """
        Get location information for a tile based on strategic locations.
        
        Args:
            filename: Tile filename
            
        Returns:
            Location info dict or None
        """
        # Check if we have cached location info
        if filename in self._tile_location_map:
            return self._tile_location_map[filename]
        
        return None
    
    def map_tiles_to_locations(self, bbox: Optional[tuple] = None) -> None:
        """
        Map tiles to strategic locations based on bounding boxes.
        
        Args:
            bbox: Optional bounding box to filter locations
        """
        logger.info("Mapping tiles to strategic locations...")
        
        for location_name, location_data in STRATEGIC_LOCATIONS.items():
            loc_bbox = location_data.get("bbox")
            if not loc_bbox:
                continue
            
            # If a bbox filter is provided, check if location overlaps
            if bbox:
                # Simple overlap check
                if (loc_bbox[2] < bbox[0] or loc_bbox[0] > bbox[2] or
                    loc_bbox[3] < bbox[1] or loc_bbox[1] > bbox[3]):
                    continue
            
            # Get tiles for this location
            tiles = self.get_tiles_in_bbox(*loc_bbox)
            
            for tile_x, tile_y in tiles:
                tile_filename = self.tile_coords_to_filename(tile_x, tile_y)
                
                self._tile_location_map[tile_filename] = {
                    "location_name": location_name,
                    "category": location_data.get("category", "general"),
                    "characteristics": location_data.get("characteristics", []),
                    "bbox": loc_bbox,
                    "description": self._generate_description(location_name, location_data)
                }
        
        logger.info(f"Mapped {len(self._tile_location_map)} tiles to locations")
    
    def _generate_description(self, location_name: str, location_data: Dict) -> str:
        """
        Generate a human-readable description for a location.
        
        Args:
            location_name: Name of the location
            location_data: Location data dictionary
            
        Returns:
            Description string
        """
        category = location_data.get("category", "general")
        characteristics = location_data.get("characteristics", [])
        
        # Format location name
        formatted_name = location_name.replace("_", " ").title()
        
        # Generate description based on category
        category_descriptions = {
            "heritage_palace": f"Historic palace area at {formatted_name}",
            "heritage_religious": f"Historic religious site at {formatted_name}",
            "heritage_fortress": f"Historic fortress at {formatted_name}",
            "urban_dense": f"Dense urban area in {formatted_name}",
            "urban_modern": f"Modern urban development in {formatted_name}",
            "coastal_urban": f"Coastal urban area near {formatted_name}",
            "suburban_residential": f"Suburban residential area in {formatted_name}",
            "infrastructure_airport": f"Airport infrastructure at {formatted_name}",
            "infrastructure_port": f"Port infrastructure at {formatted_name}",
        }
        
        description = category_descriptions.get(category, f"Area at {formatted_name}")
        
        if characteristics:
            char_str = ", ".join(characteristics[:3])
            description += f". Features: {char_str}"
        
        return description
    
    def download_tile(self, filename: str, force: bool = False,
                      subdirectory: Optional[str] = None,
                      save_tile_metadata: bool = True,
                      skip_existing: bool = True) -> Tuple[bool, bool]:
        """
        Download a single tile.
        
        Args:
            filename: IGN LiDAR HD filename
            force: Force download even if file exists (overrides skip_existing)
            subdirectory: Optional subdirectory within output_dir
            save_tile_metadata: Whether to save tile metadata JSON
            skip_existing: Skip download if tile already exists (default: True)
            
        Returns:
            Tuple of (success, was_skipped)
            - success: True if download successful or file already exists
            - was_skipped: True if download was skipped (file already present)
        """
        # Assure une extension .copc.laz si le WFS renvoie le nom sans suffixe
        save_filename = filename
        if not save_filename.lower().endswith((".laz", ".las")):
            save_filename = f"{save_filename}.copc.laz"

        # Determine output path with optional subdirectory
        if subdirectory:
            output_dir = self.output_dir / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / save_filename
        else:
            output_path = self.output_dir / save_filename
        
        # Skip if file exists and not forcing
        if output_path.exists() and skip_existing and not force:
            file_size_mb = output_path.stat().st_size // (1024*1024)
            logger.info(
                f"⏭️  {filename} already exists ({file_size_mb} MB), skipping"
            )
            return True, True
        
        # Get download URL from WFS service
        tile_name = filename.replace('.laz', '').replace('.LAZ', '')
        download_url = self.get_tile_download_url(tile_name)
        
        if not download_url:
            logger.error(f"Could not find download URL for {filename}")
            return False, False
        
        try:
            logger.info(f"Downloading {filename}...")
            
            # Make request with streaming for large files
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Write file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size_mb = output_path.stat().st_size // (1024*1024)
            logger.info(f"✓ Downloaded {filename} ({file_size_mb} MB)")
            
            # Save tile metadata if requested
            if save_tile_metadata:
                location_info = self._get_tile_location_info(filename)
                
                if location_info:
                    tile_metadata = self.metadata_manager.create_tile_metadata(
                        filename=filename,
                        location_name=location_info.get("location_name"),
                        category=location_info.get("category"),
                        characteristics=location_info.get("characteristics"),
                        description=location_info.get("description"),
                        bbox=location_info.get("bbox")
                    )
                    
                    self.metadata_manager.save_tile_metadata(
                        tile_metadata,
                        subdirectory=subdirectory
                    )
            
            return True, False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            return False, False
    
    def batch_download(self, tile_list: List[str],
                       max_retries: int = 3,
                       num_workers: int = None,
                       bbox: Optional[tuple] = None,
                       save_metadata: bool = True,
                       save_tile_metadata: bool = True,
                       skip_existing: bool = True) -> Dict[str, bool]:
        """
        Download multiple tiles with retry logic and optional parallelism.
        
        Args:
            tile_list: List of tile filenames to download
            max_retries: Maximum retry attempts per tile
            num_workers: Number of parallel download workers
                        (default: max_concurrent)
            bbox: Optional bounding box for metadata
            save_metadata: Whether to save stats.json
            save_tile_metadata: Whether to save individual tile metadata
            skip_existing: Skip tiles that already exist in output directory
            
        Returns:
            Dictionary mapping filename to success status
        """
        if num_workers is None:
            num_workers = self.max_concurrent
        
        # Map tiles to strategic locations before downloading
        if save_tile_metadata:
            self.map_tiles_to_locations(bbox)
        
        # Séquentiel si 1 worker
        if num_workers <= 1:
            results, stats = self._batch_download_sequential(
                tile_list, max_retries, skip_existing
            )
        else:
            # Parallèle avec ThreadPoolExecutor
            results, stats = self._batch_download_parallel(
                tile_list, max_retries, num_workers, skip_existing
            )
        
        # Log summary
        logger.info("")
        logger.info("="*70)
        logger.info("📊 Download Summary:")
        logger.info(f"  Total tiles requested: {len(tile_list)}")
        logger.info(f"  ✅ Successfully downloaded: {stats['downloaded']}")
        logger.info(f"  ⏭️  Skipped (already present): {stats['skipped']}")
        logger.info(f"  ❌ Failed: {stats['failed']}")
        logger.info("="*70)
        
        # Save metadata if requested
        if save_metadata:
            tiles_info = [{"filename": fn} for fn in tile_list]
            metadata_stats = self.metadata_manager.create_download_stats(
                tiles_info=tiles_info,
                bbox=bbox,
                download_results=results
            )
            metadata_stats.update(stats)
            self.metadata_manager.save_stats(metadata_stats)
        
        return results
    
    def _batch_download_sequential(
            self, tile_list: List[str],
            max_retries: int,
            skip_existing: bool = True):
        """Téléchargement séquentiel avec statistiques."""
        results = {}
        downloaded = 0
        skipped = 0
        failed = 0
        
        for filename in tile_list:
            success = False
            was_skipped = False
            
            for attempt in range(max_retries):
                if attempt > 0:
                    logger.info(
                        f"Retrying {filename} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                success, was_skipped = self.download_tile(
                    filename, skip_existing=skip_existing
                )
                if success:
                    break
            
            results[filename] = success
            
            if success and was_skipped:
                skipped += 1
            elif success:
                downloaded += 1
            else:
                failed += 1
            
            # Brief pause between downloads to be respectful
            if not was_skipped:
                time.sleep(1)
        
        stats = {
            'downloaded': downloaded,
            'skipped': skipped,
            'failed': failed
        }
        
        return results, stats
    
    def _batch_download_parallel(
            self, tile_list: List[str],
            max_retries: int,
            num_workers: int,
            skip_existing: bool = True):
        """
        Téléchargement parallèle avec ThreadPoolExecutor et statistiques.
        
        Args:
            tile_list: Liste des fichiers à télécharger
            max_retries: Nombre de tentatives maximum
            num_workers: Nombre de threads parallèles
            skip_existing: Skip tiles that already exist
            
        Returns:
            Tuple of (results dict, stats dict)
        """
        logger.info(f"🚀 Téléchargement parallèle avec {num_workers} workers")
        results = {}
        downloaded = 0
        skipped = 0
        failed = 0
        
        def download_with_retry(filename: str):
            """Télécharger un fichier avec retry."""
            for attempt in range(max_retries):
                if attempt > 0:
                    logger.debug(
                        f"Retry {filename} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(2 ** attempt)
                
                success, was_skipped = self.download_tile(
                    filename, skip_existing=skip_existing
                )
                if success:
                    return (filename, True, was_skipped)
            
            return (filename, False, False)
        
        # Utiliser ThreadPoolExecutor pour parallélisme I/O
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Soumettre toutes les tâches
            futures = {
                executor.submit(download_with_retry, filename): filename
                for filename in tile_list
            }
            
            # Collecter les résultats avec barre de progression
            with tqdm(total=len(tile_list), desc="Téléchargement") as pbar:
                for future in as_completed(futures):
                    filename, success, was_skipped = future.result()
                    results[filename] = success
                    
                    if success and was_skipped:
                        skipped += 1
                    elif success:
                        downloaded += 1
                    else:
                        failed += 1
                        logger.warning(f"❌ Échec: {filename}")
                    
                    pbar.update(1)
        
        stats = {
            'downloaded': downloaded,
            'skipped': skipped,
            'failed': failed
        }
        
        return results, stats
