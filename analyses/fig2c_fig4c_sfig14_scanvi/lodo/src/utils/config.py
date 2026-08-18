"""
Configuration management for multi-lineage benchmarking.

Loads and validates YAML config files for each lineage.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(lineage: str, config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load configuration for a specific lineage.
    
    Args:
        lineage: Lineage name (myeloid, lymphoid, epithelial, stroma)
        config_dir: Directory containing config files
        
    Returns:
        Dictionary with complete configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Resolve configs directory to an absolute path relative to the repo root,
    # so notebooks can call this from any CWD
    module_root = Path(__file__).resolve().parents[2]  # .../reference_mapping_benchmark
    config_dir_abs = (module_root / config_dir) if not Path(config_dir).is_absolute() else Path(config_dir)

    config_path = config_dir_abs / f"{lineage}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {list(config_dir_abs.glob('*.yaml'))}"
        )
    
    logger.info(f"Loading config for {lineage} from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = _resolve_data_paths(config, lineage)
    validate_config(config, lineage)
    config = _substitute_lineage(config, lineage)
    
    logger.info(f"✅ Config loaded successfully for {lineage}")
    return config


def validate_config(config: Dict[str, Any], lineage: str) -> None:
    """
    Validate that config has all required fields.
    
    Args:
        config: Configuration dictionary
        lineage: Lineage name for error messages
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = {
        'lineage': str,
        'display_name': str,
        'data': dict,
        'labels': list,
        'scanvi': dict,
        'scimilarity': dict,
        'output': dict,
        'taxonomy': dict
    }
    
    for field, field_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {lineage} config")
        if not isinstance(config[field], field_type):
            raise ValueError(
                f"Field '{field}' must be {field_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )
    
    # Validate data paths exist (warn only — configs often point at cluster paths)
    data_path = Path(config['data']['input_path'])
    if not data_path.exists():
        logger.warning("Data path does not exist (yet): %s", data_path)
    
    # Validate taxonomy path if provided
    if config['taxonomy'].get('path'):
        tax_path = Path(config['taxonomy']['path'])
        if not tax_path.exists():
            logger.warning(f"Taxonomy path does not exist: {tax_path}")
    
    logger.info(f"✅ Config validation passed for {lineage}")


def _resolve_data_paths(config: Dict[str, Any], lineage: str) -> Dict[str, Any]:
    """Replace cluster placeholders with HGCA_OBJECTS / PANGI_H5AD when set."""
    data = config.setdefault("data", {})
    current = str(data.get("input_path") or "")
    cluster = current.startswith("/lab-share") or current == ""
    objects = os.environ.get("HGCA_OBJECTS", "")
    pangi = os.environ.get("PANGI_H5AD", "")
    slug = lineage.replace("pangi_", "")
    if cluster and lineage.startswith("pangi") and pangi:
        data["input_path"] = pangi
    elif cluster and objects:
        data["input_path"] = str(Path(objects) / f"{slug}.h5ad")
    scim = config.setdefault("scimilarity", {})
    model_dir = str(scim.get("model_dir") or "")
    if model_dir.startswith("/lab-share"):
        scim["model_dir"] = os.environ.get("SCIMILARITY_MODEL_DIR", "")
    return config


def _substitute_lineage(config: Dict[str, Any], lineage: str) -> Dict[str, Any]:
    """
    Substitute {lineage} placeholders in config paths.
    
    Args:
        config: Configuration dictionary
        lineage: Lineage name
        
    Returns:
        Config with substituted paths
    """
    # Substitute in output paths
    if 'output' in config:
        if 'results_dir' in config['output']:
            config['output']['results_dir'] = config['output']['results_dir'].format(lineage=lineage)
        if 'plots_dir' in config['output']:
            config['output']['plots_dir'] = config['output']['plots_dir'].format(lineage=lineage)
    
    return config


def get_available_lineages(config_dir: str = "configs") -> list:
    """
    Get list of available lineage configs.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        List of lineage names
    """
    module_root = Path(__file__).resolve().parents[2]
    config_path = (module_root / config_dir) if not Path(config_dir).is_absolute() else Path(config_dir)
    if not config_path.exists():
        return []
    
    configs = list(config_path.glob("*.yaml"))
    # Exclude base_config.yaml if it exists
    lineages = [c.stem for c in configs if c.stem != 'base_config']
    
    return sorted(lineages)


# Example usage:
if __name__ == "__main__":
    # Test config loading
    for lineage in ["myeloid", "lymphoid", "epithelial", "stroma"]:
        try:
            config = load_config(lineage)
            print(f"\n✅ {lineage.upper()}")
            print(f"   Display name: {config['display_name']}")
            print(f"   Data path: {config['data']['input_path']}")
            print(f"   Results dir: {config['output']['results_dir']}")
        except FileNotFoundError as e:
            print(f"\n⚠️  {lineage.upper()}: Config not created yet")
        except Exception as e:
            print(f"\n❌ {lineage.upper()}: {e}")

