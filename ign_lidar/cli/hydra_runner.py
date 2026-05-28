"""
Hydra runner utility for programmatic Hydra usage.

Enables using Hydra configuration system within Click commands without
decorator-based entry points. This provides flexibility to integrate
Hydra configuration management with Click's command structure.

Usage:
    from ign_lidar.cli.hydra_runner import HydraRunner
    
    runner = HydraRunner()
    cfg = runner.load_config(
        config_name="config",
        overrides=["processor.use_gpu=true", "input_dir=data/"]
    )
    
    # Use config...
    processor = LiDARProcessor(config=cfg)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Check Hydra availability
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    logger.warning("Hydra not available. Install with: pip install hydra-core")


class HydraRunner:
    """
    Programmatic Hydra configuration loader.
    
    This class wraps Hydra's compose functionality for use in non-decorator
    contexts, enabling integration with Click commands.
    
    Examples:
        # Load default config
        runner = HydraRunner()
        cfg = runner.load_config()
        
        # Load with overrides
        cfg = runner.load_config(overrides=["processor.use_gpu=true"])
        
        # Load specific file
        cfg = runner.load_config(config_file="examples/config_gpu.yaml")
        
        # Mix file and overrides
        cfg = runner.load_config(
            config_file="config.yaml",
            overrides=["input_dir=data/"]
        )
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize Hydra runner.
        
        Args:
            config_dir: Path to configuration directory. If None, uses
                       package's configs/ directory.
        """
        if not HYDRA_AVAILABLE:
            raise ImportError(
                "Hydra is not installed. Install with: pip install hydra-core omegaconf"
            )
        
        if config_dir is None:
            # Default to package configs directory
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir).absolute()
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        logger.debug(f"HydraRunner initialized with config_dir: {self.config_dir}")
    
    def load_config(
        self,
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        return_dict: bool = False
    ) -> Union[DictConfig, Dict[str, Any]]:
        """
        Load Hydra configuration with optional overrides.
        
        Args:
            config_name: Base config name (without .yaml extension).
                        Used when loading from config_dir. Default: "config"
            overrides: List of Hydra-style overrides.
                      Examples: ["processor.use_gpu=true", "input_dir=data/"]
            config_file: Path to a specific config file to load.
                        If provided, loads this file instead of using config_name.
            return_dict: If True, returns plain dict instead of DictConfig.
            
        Returns:
            Configuration object (DictConfig or dict)
            
        Examples:
            # Load default config with overrides
            cfg = runner.load_config(overrides=["processor.use_gpu=true"])
            
            # Load specific file
            cfg = runner.load_config(config_file="examples/config_gpu.yaml")
            
            # Mix both
            cfg = runner.load_config(
                config_file="config.yaml",
                overrides=["input_dir=data/", "processor.use_gpu=true"]
            )
        """
        overrides = overrides or []
        
        if config_file:
            # Load specific file directly
            config_file = Path(config_file)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            logger.debug(f"Loading config from file: {config_file}")
            
            # First load the base config to get defaults
            base_config_path = self.config_dir / f"{config_name}.yaml"
            if base_config_path.exists():
                logger.debug(f"Loading base config from: {base_config_path}")
                base_cfg = OmegaConf.load(base_config_path)
            else:
                # If base config doesn't exist, create minimal defaults
                logger.debug("No base config found, using minimal defaults")
                base_cfg = OmegaConf.create({
                    'bbox': {
                        'xmin': None,
                        'ymin': None,
                        'xmax': None,
                        'ymax': None
                    }
                })
            
            # Load user config
            user_cfg = OmegaConf.load(config_file)
            
            # Handle Hydra 'defaults' section if present
            if 'defaults' in user_cfg:
                defaults = user_cfg.get('defaults', [])
                logger.debug(f"Processing {len(defaults)} defaults from config file")
                
                # Load each default config and merge
                for default in defaults:
                    if default == '_self_':
                        continue  # Skip _self_ marker
                    
                    # Handle different default formats
                    if isinstance(default, str):
                        if default.startswith('../'):
                            default_path = self.config_dir / f"{default.replace('../', '')}.yaml"
                        else:
                            default_path = self.config_dir / f"{default}.yaml"
                    elif isinstance(default, dict):
                        # Handle override syntax like {preset: lod3}
                        for key, value in default.items():
                            default_path = self.config_dir / key / f"{value}.yaml"
                            break
                    else:
                        continue
                    
                    if default_path.exists():
                        logger.debug(f"  Loading default: {default_path}")
                        default_cfg = OmegaConf.load(default_path)
                        
                        # Recursively handle defaults in the included config
                        if 'defaults' in default_cfg:
                            nested_defaults = default_cfg.pop('defaults', [])
                            for nested_default in nested_defaults:
                                if nested_default == '_self_':
                                    continue
                                if isinstance(nested_default, str) and nested_default.startswith('../'):
                                    # Handle relative paths like ../base
                                    nested_path = self.config_dir / nested_default.replace('../', '') / "base.yaml"
                                    if not nested_path.exists():
                                        nested_path = self.config_dir / f"{nested_default.replace('../', '')}.yaml"
                                    if nested_path.exists():
                                        logger.debug(f"    Loading nested default: {nested_path}")
                                        nested_cfg = OmegaConf.load(nested_path)
                                        base_cfg = OmegaConf.merge(base_cfg, nested_cfg)
                        
                        base_cfg = OmegaConf.merge(base_cfg, default_cfg)
                    else:
                        logger.warning(f"  Default config not found: {default_path}")
                
                # Remove defaults from user config before merging
                user_cfg = OmegaConf.to_container(user_cfg)
                user_cfg.pop('defaults', None)
                user_cfg = OmegaConf.create(user_cfg)
            
            # Merge: base config < defaults < user config < overrides
            cfg = OmegaConf.merge(base_cfg, user_cfg)
            
            # Apply overrides
            if overrides:
                logger.debug(f"Applying {len(overrides)} overrides")
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
        else:
            # Use Hydra compose with config directory
            logger.debug(f"Loading config '{config_name}' from {self.config_dir}")
            
            # Clean up any existing Hydra instance
            GlobalHydra.instance().clear()
            
            # Initialize and compose
            try:
                with initialize_config_dir(
                    config_dir=str(self.config_dir),
                    version_base=None,
                    job_name="ign_lidar_cli"
                ):
                    cfg = compose(
                        config_name=config_name,
                        overrides=overrides
                    )
                    logger.debug(f"Config loaded with {len(overrides)} overrides")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                raise
        
        # Convert to dict if requested
        if return_dict:
            return OmegaConf.to_container(cfg, resolve=True)
        
        return cfg
    
    @staticmethod
    def merge_configs(
        base_cfg: DictConfig,
        *override_cfgs: DictConfig
    ) -> DictConfig:
        """
        Merge multiple configurations.
        
        Later configs override earlier ones.
        
        Args:
            base_cfg: Base configuration
            *override_cfgs: Additional configs to merge
            
        Returns:
            Merged configuration
            
        Example:
            base = runner.load_config("base_config")
            gpu = runner.load_config("gpu_config")
            merged = HydraRunner.merge_configs(base, gpu)
        """
        result = base_cfg
        for override_cfg in override_cfgs:
            result = OmegaConf.merge(result, override_cfg)
        return result
    
    @staticmethod
    def validate_config(cfg: DictConfig, schema_cls: Optional[type] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            cfg: Configuration to validate
            schema_cls: Optional schema class to validate against
            
        Returns:
            True if valid, raises exception otherwise
            
        Example:
            from ign_lidar.config.schema import IGNLiDARConfig
            cfg = runner.load_config()
            HydraRunner.validate_config(cfg, IGNLiDARConfig)
        """
        try:
            if schema_cls:
                # Validate against structured config
                structured = OmegaConf.to_object(cfg)
                if not isinstance(structured, schema_cls):
                    raise ValueError(f"Config does not match schema {schema_cls.__name__}")
                
                # Check for validation method
                if hasattr(structured, 'validate'):
                    structured.validate()
            
            # Check for required fields (basic validation)
            required_fields = ['input_dir', 'output_dir']
            missing = [f for f in required_fields if f not in cfg]
            
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    @staticmethod
    def print_config(cfg: DictConfig, title: str = "Configuration") -> None:
        """
        Print configuration in a readable format.
        
        Args:
            cfg: Configuration to print
            title: Optional title for the output
            
        Example:
            cfg = runner.load_config()
            HydraRunner.print_config(cfg, "Processing Configuration")
        """
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 70)
    
    @staticmethod
    def extract_overrides_from_args(args: List[str]) -> tuple[List[str], List[str]]:
        """
        Separate Hydra overrides from other arguments.
        
        Hydra overrides contain '=' and don't start with '-'.
        
        Args:
            args: List of command-line arguments
            
        Returns:
            Tuple of (regular_args, hydra_overrides)
            
        Example:
            args = ['--verbose', 'input_dir=data/', 'processor.use_gpu=true']
            regular, overrides = HydraRunner.extract_overrides_from_args(args)
            # regular = ['--verbose']
            # overrides = ['input_dir=data/', 'processor.use_gpu=true']
        """
        regular_args = []
        hydra_overrides = []
        
        for arg in args:
            if '=' in arg and not arg.startswith('-'):
                hydra_overrides.append(arg)
            else:
                regular_args.append(arg)
        
        return regular_args, hydra_overrides


# Convenience function for simple use cases
def load_config(
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
    config_name: str = "config"
) -> DictConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Path to config file (optional)
        overrides: List of overrides (optional)
        config_name: Config name if not using file (default: "config")
        
    Returns:
        Loaded configuration
        
    Example:
        # Simple usage
        cfg = load_config(overrides=["processor.use_gpu=true"])
        
        # From file
        cfg = load_config(config_file="examples/config_gpu.yaml")
    """
    runner = HydraRunner()
    return runner.load_config(
        config_name=config_name,
        overrides=overrides,
        config_file=config_file
    )


__all__ = [
    'HydraRunner',
    'load_config',
    'HYDRA_AVAILABLE'
]
