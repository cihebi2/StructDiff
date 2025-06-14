import torch
import numpy as np
import random
import os
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import hashlib
import json


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get torch device"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if isinstance(device, str):
        device = torch.device(device)
    
    return device


def move_to_device(
    data: Union[torch.Tensor, Dict, List],
    device: torch.device
) -> Union[torch.Tensor, Dict, List]:
    """Recursively move data to device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def compute_gradient_norm(
    model: torch.nn.Module,
    norm_type: float = 2.0
) -> float:
    """Compute gradient norm of model parameters"""
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1. / norm_type)
    
    return total_norm


def get_experiment_id(config: Dict) -> str:
    """Generate unique experiment ID from config"""
    # Create a hash of important config parameters
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    
    return hash_obj.hexdigest()[:8]


def create_exp_name(
    base_name: str,
    config: Dict,
    timestamp: bool = True
) -> str:
    """Create experiment name from config"""
    parts = [base_name]
    
    # Add important hyperparameters
    if 'model' in config:
        parts.append(f"h{config['model'].get('hidden_dim', 'NA')}")
    
    if 'training' in config:
        parts.append(f"lr{config['training'].get('lr', 'NA')}")
    
    if 'diffusion' in config:
        parts.append(f"t{config['diffusion'].get('num_timesteps', 'NA')}")
    
    # Add timestamp
    if timestamp:
        from datetime import datetime
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    return "_".join(map(str, parts))


def save_metrics(
    metrics: Dict[str, float],
    save_path: Union[str, Path],
    epoch: Optional[int] = None,
    append: bool = True
):
    """Save metrics to JSON file"""
    save_path = Path(save_path)
    
    # Add epoch if provided
    if epoch is not None:
        metrics = {'epoch': epoch, **metrics}
    
    # Load existing metrics if appending
    if append and save_path.exists():
        with open(save_path, 'r') as f:
            existing = json.load(f)
        
        if isinstance(existing, list):
            existing.append(metrics)
            metrics = existing
        else:
            metrics = [existing, metrics]
    else:
        metrics = [metrics] if append else metrics
    
    # Save metrics
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(
    load_path: Union[str, Path]
) -> Union[Dict, List[Dict]]:
    """Load metrics from JSON file"""
    with open(load_path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_batch_size(batch: Union[torch.Tensor, Dict]) -> int:
    """Get batch size from batch data"""
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    elif isinstance(batch, dict):
        # Get from first tensor in dict
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
    
    raise ValueError("Cannot determine batch size")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """Compute mean of tensor with mask"""
    if dim is not None:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)
    else:
        return (tensor * mask).sum() / mask.sum().clamp(min=1e-8)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    
    Args:
        logits: logits distribution shape (batch_size, vocab_size)
        top_k: keep only top k tokens with highest probability
        top_p: keep the top tokens with cumulative probability >= top_p
        temperature: temperature for sampling
        
    Returns:
        Filtered logits
    """
    # Adjust logits with temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        # Remove all tokens with a probability less than the k-th highest
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    return logits


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self, name: str = "Meter"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} ({self.avg:.4f})"
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
