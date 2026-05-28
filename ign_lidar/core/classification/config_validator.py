"""
Configuration Validator Module

Validates and normalizes processor configuration.
"""

import logging
from typing import List, Dict, Optional, Literal
from pathlib import Path

logger = logging.getLogger(__name__)

# Type definitions
ProcessingMode = Literal["patches_only", "both", "enriched_only", "reclassify_only"]


class ConfigValidator:
    """
    Validates and normalizes LiDAR processor configuration.
    
    Handles:
    - Output format validation
    - Processing mode validation  
    - Preprocessing config setup
    - Stitching config setup
    - Reclassification mode
    """
    
    SUPPORTED_FORMATS = ['npz', 'hdf5', 'pytorch', 'torch', 'laz', 'ptv3_pointcept']
    VALID_PROCESSING_MODES = [
        "patches_only", "both", "enriched_only", "reclassify_only",
        "enriched_and_patches",
    ]
    
    @staticmethod
    def validate_output_format(output_format: str) -> List[str]:
        """
        Validate output format(s).
        
        Supports comma-separated multi-format: 'hdf5,laz'
        
        Args:
            output_format: Format string (single or comma-separated)
            
        Returns:
            List of validated format strings
            
        Raises:
            ValueError: If format is not supported
        """
        formats_list = [fmt.strip() for fmt in output_format.split(',')]
        
        for fmt in formats_list:
            if fmt not in ConfigValidator.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported output format: '{fmt}'. "
                    f"Supported formats: {', '.join(ConfigValidator.SUPPORTED_FORMATS)}\n"
                    f"For multiple formats, use comma-separated list: 'hdf5,laz'"
                )
        
        return formats_list
    
    @staticmethod
    def validate_processing_mode(mode: str) -> ProcessingMode:
        """
        Validate processing mode.
        
        Args:
            mode: Processing mode string
            
        Returns:
            Validated processing mode
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ConfigValidator.VALID_PROCESSING_MODES:
            raise ValueError(
                f"Invalid processing mode: '{mode}'. "
                f"Valid modes: {', '.join(ConfigValidator.VALID_PROCESSING_MODES)}"
            )
        
        return mode
    
    @staticmethod
    def check_pytorch_availability(formats_list: List[str]) -> None:
        """
        Check if PyTorch is available when torch format is requested.
        
        Args:
            formats_list: List of requested formats
            
        Raises:
            ImportError: If PyTorch format requested but torch not installed
        """
        if any(fmt in ['pytorch', 'torch'] for fmt in formats_list):
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "PyTorch format requested but torch is not installed. "
                    "Install with: pip install torch"
                )
    
    @staticmethod
    def setup_preprocessing_config(
        preprocess: bool,
        preprocess_config: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Setup preprocessing configuration with defaults.
        
        Args:
            preprocess: Whether preprocessing is enabled
            preprocess_config: Custom preprocessing configuration
            
        Returns:
            Preprocessing configuration dict or None
        """
        if not preprocess:
            return None
        
        if preprocess_config is None:
            # Sensible defaults
            return {
                'sor_enabled': True,
                'sor_k': 12,
                'sor_std': 2.0,
                'ror_enabled': True,
                'ror_radius': 1.0,
                'ror_neighbors': 4,
                'voxel_enabled': False,
                'voxel_size': 0.1
            }
        
        return preprocess_config.copy()
    
    @staticmethod
    def setup_stitching_config(
        use_stitching: bool,
        buffer_size: float = 10.0,
        stitching_config: Optional[Dict] = None
    ) -> Dict:
        """
        Setup stitching configuration with defaults.
        
        Args:
            use_stitching: Whether stitching is enabled
            buffer_size: Buffer zone size in meters
            stitching_config: Custom stitching configuration
            
        Returns:
            Stitching configuration dict
        """
        if stitching_config is None:
            config = {
                'enabled': use_stitching,
                'buffer_size': buffer_size,
                'auto_detect_neighbors': True,
                'auto_download_neighbors': False,
                'cache_enabled': True
            }
        else:
            config = stitching_config.copy()
            # Override enable flag and buffer size
            config['enabled'] = use_stitching
            if 'buffer_size' not in config:
                config['buffer_size'] = buffer_size
            # Default auto_download_neighbors to False if not specified
            if 'auto_download_neighbors' not in config:
                config['auto_download_neighbors'] = False
        
        return config
    
    @staticmethod
    def init_stitcher(stitching_config: Dict):
        """
        Initialize advanced tile stitcher if configured.
        
        Args:
            stitching_config: Stitching configuration
            
        Returns:
            TileStitcher instance or None
        """
        if not stitching_config.get('enabled', False):
            return None
        
        if not stitching_config.get('use_stitcher', False):
            return None
        
        try:
            from ..tile_stitcher import TileStitcher
            stitcher = TileStitcher(config=stitching_config)
            logger.info("Advanced tile stitcher initialized")
            return stitcher
        except ImportError as e:
            logger.warning(f"Advanced stitcher unavailable: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize stitcher: {e}")
            return None
