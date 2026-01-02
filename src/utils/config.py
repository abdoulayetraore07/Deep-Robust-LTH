"""
Configuration utilities for loading and validating config.yaml
"""

import yaml
import os
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Loads and validates configuration from YAML file
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'data.n_train')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config = ConfigLoader()
            >>> n_train = config.get('data.n_train')
            >>> lr = config.get('training.learning_rate', 0.001)
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'data.n_train')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
    
    def save(self, save_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            save_path: Path to save config (default: original path)
        """
        path = save_path or self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dict-like access: config['data']
        """
        return self.config[key]
    
    def __repr__(self) -> str:
        """
        String representation
        """
        return f"ConfigLoader(config_path='{self.config_path}')"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Convenience function to load config directly
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration has all required fields
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required_sections = [
        'data',
        'model',
        'training',
        'pruning',
        'attacks',
        'adversarial_training',
        'evaluation',
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data section
    if 'heston' not in config['data']:
        raise ValueError("Missing 'heston' parameters in data config")
    
    # Validate model section
    if 'hidden_dims' not in config['model']:
        raise ValueError("Missing 'hidden_dims' in model config")
    
    # Validate training section
    if 'learning_rate' not in config['training']:
        raise ValueError("Missing 'learning_rate' in training config")


def get_device(config: Dict[str, Any]) -> str:
    """
    Get device from config with automatic detection
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    import torch
    
    device_config = config.get('compute', {}).get('device', 'auto')
    
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Config] Auto-detected device: {device}")
    else:
        device = device_config
        if device == 'cuda' and not torch.cuda.is_available():
            print("[Config] WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    return device


# ============================================================================
# NOUVELLES FONCTIONS POUR CACHING ET CHECKPOINTING
# ============================================================================

def compute_config_hash(config: Dict[str, Any], include_keys: Optional[list] = None) -> str:
    """
    Compute a unique hash based on critical config parameters.
    This hash identifies a specific experiment configuration.
    
    Args:
        config: Configuration dictionary
        include_keys: List of top-level keys to include in hash.
                     If None, uses ['data', 'model', 'training']
    
    Returns:
        8-character hex hash string
    """
    if include_keys is None:
        # Only hash parameters that affect the model/training
        include_keys = ['data', 'model', 'training']
    
    # Extract relevant config sections
    hash_dict = {}
    for key in include_keys:
        if key in config:
            hash_dict[key] = config[key]
    
    # Convert to stable JSON string (sorted keys)
    json_str = json.dumps(hash_dict, sort_keys=True, default=str)
    
    # Compute SHA256 hash and take first 8 characters
    hash_obj = hashlib.sha256(json_str.encode())
    return hash_obj.hexdigest()[:8]


def get_experiment_dir(config: Dict[str, Any], base_dir: str = "experiments") -> Path:
    """
    Get the experiment directory based on config hash.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    config_hash = compute_config_hash(config)
    exp_name = config.get('experiment_name', 'default')
    
    # Format: experiments/{exp_name}_{hash}/
    exp_dir = Path(base_dir) / f"{exp_name}_{config_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def get_checkpoint_path(config: Dict[str, Any], checkpoint_type: str = "best") -> Path:
    """
    Get path to checkpoint file.
    
    Args:
        config: Configuration dictionary
        checkpoint_type: 'best', 'latest', or 'init'
        
    Returns:
        Path to checkpoint file
    """
    exp_dir = get_experiment_dir(config)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_type == "best":
        return checkpoint_dir / "best_model.pt"
    elif checkpoint_type == "latest":
        return checkpoint_dir / "latest_checkpoint.pt"
    elif checkpoint_type == "init":
        return checkpoint_dir / "init_weights.pt"
    else:
        return checkpoint_dir / f"{checkpoint_type}.pt"


def check_existing_model(config: Dict[str, Any], model_type: str = "baseline") -> Dict[str, Any]:
    """
    Check if a trained model already exists for this configuration.
    
    Args:
        config: Configuration dictionary
        model_type: 'baseline', 'pruned', 'adversarial'
        
    Returns:
        Dictionary with:
            - exists: bool
            - checkpoint_path: Path or None
            - training_complete: bool
            - last_epoch: int or None
    """
    exp_dir = get_experiment_dir(config)
    
    result = {
        'exists': False,
        'checkpoint_path': None,
        'training_complete': False,
        'last_epoch': None,
        'exp_dir': exp_dir
    }
    
    # Check for best model (training complete)
    best_path = exp_dir / "checkpoints" / f"{model_type}_best.pt"
    if best_path.exists():
        result['exists'] = True
        result['checkpoint_path'] = best_path
        result['training_complete'] = True
        return result
    
    # Check for latest checkpoint (training incomplete)
    latest_path = exp_dir / "checkpoints" / f"{model_type}_latest.pt"
    if latest_path.exists():
        import torch
        checkpoint = torch.load(latest_path, map_location='cpu')
        result['exists'] = True
        result['checkpoint_path'] = latest_path
        result['training_complete'] = False
        result['last_epoch'] = checkpoint.get('epoch', 0)
        return result
    
    return result


def should_skip_training(config: Dict[str, Any], model_type: str = "baseline", 
                         force_retrain: bool = False) -> bool:
    """
    Determine if training should be skipped (model already exists).
    
    Args:
        config: Configuration dictionary
        model_type: Type of model to check
        force_retrain: If True, never skip
        
    Returns:
        True if training should be skipped
    """
    if force_retrain:
        return False
    
    cache_mode = config.get('caching', {}).get('mode', 'on')
    
    if cache_mode == 'off':
        return False
    
    existing = check_existing_model(config, model_type)
    
    if existing['training_complete']:
        print(f"[Config] Found existing trained model at {existing['checkpoint_path']}")
        print(f"[Config] Skipping training. Use force_retrain=True to override.")
        return True
    
    return False


def save_config_to_experiment(config: Dict[str, Any]) -> None:
    """
    Save a copy of the config to the experiment directory.
    
    Args:
        config: Configuration dictionary
    """
    exp_dir = get_experiment_dir(config)
    config_path = exp_dir / "config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Also save the hash for reference
    hash_path = exp_dir / "config_hash.txt"
    with open(hash_path, 'w') as f:
        f.write(compute_config_hash(config))


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")