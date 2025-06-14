import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import torch


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        overrides: Dictionary of values to override
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Load any included configs
    if 'includes' in config:
        for include_path in config.includes:
            include_config = OmegaConf.load(include_path)
            config = OmegaConf.merge(config, include_config)
    
    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    # Resolve any interpolations
    OmegaConf.resolve(config)
    
    return config


def save_config(
    config: Union[Dict, DictConfig],
    save_path: Union[str, Path]
):
    """Save configuration to file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(
    *configs: Union[Dict, DictConfig]
) -> DictConfig:
    """Merge multiple configurations"""
    merged = OmegaConf.create({})
    
    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        merged = OmegaConf.merge(merged, config)
    
    return merged


def update_config_with_args(
    config: DictConfig,
    args: Any
) -> DictConfig:
    """Update config with command line arguments"""
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    
    # Remove None values
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    
    # Convert to OmegaConf
    args_config = OmegaConf.create(args_dict)
    
    # Merge with priority to args
    return OmegaConf.merge(config, args_config)


def validate_config(config: DictConfig) -> bool:
    """Validate configuration"""
    required_fields = [
        'model.hidden_dim',
        'training.num_epochs',
        'data.train_path',
        'diffusion.num_timesteps'
    ]
    
    for field in required_fields:
        try:
            OmegaConf.select(config, field, throw_on_missing=True)
        except:
            raise ValueError(f"Required config field missing: {field}")
    
    return True


def get_model_config(config: DictConfig) -> DictConfig:
    """Extract model configuration"""
    if 'model' in config and 'config_path' in config.model:
        model_config = load_config(config.model.config_path)
        # Merge with any inline model config
        model_config = OmegaConf.merge(
            model_config,
            config.model
        )
        return model_config
    
    return config.model


def setup_experiment_dir(
    config: DictConfig,
    experiment_name: Optional[str] = None
) -> Path:
    """Setup experiment directory structure"""
    if experiment_name is None:
        experiment_name = config.get('experiment.name', 'experiment')
    
    # Create experiment directory
    exp_dir = Path(config.experiment.output_dir) / experiment_name
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'results').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'configs').mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, exp_dir / 'configs' / 'config.yaml')
    
    return exp_dir


def get_device(config: DictConfig) -> torch.device:
    """Get device from config"""
    if 'device' in config:
        device_str = config.device
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    
    return torch.device(device_str)


class ConfigManager:
    """Manage configurations throughout training"""
    
    def __init__(self, base_config_path: Union[str, Path]):
        self.base_config = load_config(base_config_path)
        self.runtime_config = OmegaConf.create({})
        
    def update_runtime(self, **kwargs):
        """Update runtime configuration"""
        update = OmegaConf.create(kwargs)
        self.runtime_config = OmegaConf.merge(self.runtime_config, update)
    
    def get_config(self) -> DictConfig:
        """Get merged configuration"""
        return OmegaConf.merge(self.base_config, self.runtime_config)
    
    def save_checkpoint_config(self, checkpoint_path: Union[str, Path]):
        """Save config alongside checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        config_path = checkpoint_path.parent / f"{checkpoint_path.stem}_config.yaml"
        save_config(self.get_config(), config_path)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
